#include "lbm_mpi.hh"
#include "mpi.h"
#include <algorithm>

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
const int LBM_MPI::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM_MPI::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM_MPI::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM_MPI::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

LBM_MPI::LBM_MPI(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r, MPI_Comm comm)
  : nx_global_(nx), ny_(ny), u_in_(u_in), tau_(0.0), comm_(comm)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);

  const std::size_t base  = nx_global_ / size_;
  const std::size_t rem = nx_global_ % size_;
  nx_local_ = base;
  if (rank_ < rem) {nx_local_ ++};
  x_start_ = rank_ * base + std::min(rem, rank_);
  nx_nbrs_ = nx_local +2;
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  f_.resize(9 * nx_nbrs_ * ny, 0.0);
  ftmp_.resize(9 * nx_nbrs_ * ny, 0.0);
  solid_.resize(nx_nbrs_ * ny, 0);

  // No-slip top and bottom walls.
  for (std::size_t x = 1; x <= nx_local_; ++x) {
    solid_[idx(x, 0)]        = 1;
    solid_[idx(x, ny_ - 1)]  = 1;
  }
  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

void
LBM_MPI::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM_MPI::mark_obstacle(double c_x, double c_y, double r)
{
  const double r2 = r * r;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 1; x <= nx_local_; ++x) {
      const std::size_t x_global = x_start_ + x - 1;
      const double dx = double(x_global) - c_x;
      const double dy = double(y) - c_y;
      if (dx * dx + dy * dy <= r2) solid_[idx(x, y)] = 1;
    }
  }
}

void
LBM_MPI::initialize()
{
  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t x = 1; x <= nx_local_; ++x) {
      const double rho = 1.0;
      const double ux  = solid_[idx(x, y)] ? 0.0 : u_in_;
      const double uy  = 0.0;
      const double u2  = ux * ux + uy * uy;
      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[fidx(i, x, y)] = feq;
      }
    }
  }
}

void
LBM_MPI::step()
{
  collide();
  bounce_back();
  exchange_nbrs();
  stream();
  apply_inlet();
  apply_outlet();
}

void
LBM_MPI::collide()
{
  const std::size_t N = nx_nbrs_ * ny_;
  const double inv_tau = 1.0 / tau_;

  for (std::size_t y = 0; y <ny_; ++y) {
    for (std::size_t x= 1; x <=nx_local_; ++x) {
      const std::size_t k = idx(x,y);
      double rho = 0.0, mx = 0.0, my = 0.0;
      for (int i = 0; i < Q; ++i) {
        const double fi = f_[i * N + k];
        rho += fi;
        mx  += cx[i] * fi;
        my  += cy[i] * fi;
      }
      const double ux = (rho > 0.0) ? mx / rho : 0.0;
      const double uy = (rho > 0.0) ? my / rho : 0.0;
      const double u2 = ux * ux + uy * uy;

      for (int i = 0; i < Q; ++i) {
        const double cu  = cx[i] * ux + cy[i] * uy;
        const double feq = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
        f_[i * N + k] += -inv_tau * (f_[i * N + k] - feq);
      }
    }
  }
}

void
LBM_MPI::bounce_back()
{
  // Fullway bounce-back: in solid cells, swap each pair of opposite directions.
  // Combined with subsequent streaming this reflects populations across the
  // solid-fluid interface.
  const std::size_t N = nx_nbrs_ * ny_;
  for (std::size_t y = 0; y< ny_; ++y) {
    for (std::size_t x = 1; y< nx_local_; ++x) {
      if (!solid_[k]) continue;
      std::swap(f_[1 * N + k], f_[3 * N + k]);
      std::swap(f_[2 * N + k], f_[4 * N + k]);
      std::swap(f_[5 * N + k], f_[7 * N + k]);
      std::swap(f_[6 * N + k], f_[8 * N + k]);
     }
  }
}

void LBM_MPI::exchange_nbrs()
{

}


void
LBM_MPI::stream()
{
  // Pull-style streaming: ftmp[i, x, y] = f[i, x - cx[i], y - cy[i]].
  // Boundary cells whose source would be outside the domain keep their
  // current value; the inlet/outlet routines overwrite the relevant ones.
  const std::size_t N = nx_nbrs_ * ny_;
  for (int i = 0; i < Q; ++i) {
    for (std::size_t y = 0; y < ny_; ++y) {
      for (std::size_t x = 1; x <= nx_local_; ++x) {
        const long sx = long(x) - cx[i];
        const long sy = long(y) - cy[i];
        if (sx >= 0 && sx < long(nx_nbrs_) && sy >= 0 && sy < long(ny_)) {
          ftmp_[i * N + idx(x, y)] = f_[i * N + idx(std::size_t(sx), std::size_t(sy))];
        } else {
          ftmp_[i * N + idx(x, y)] = f_[i * N + idx(x, y)];
        }
      }
    }
  }
  f_.swap(ftmp_);
}

void
LBM_MPI::apply_inlet()
{
  // Reset the inlet column to the equilibrium distribution corresponding
  // to a prescribed uniform velocity (u_in, 0) and unit density. Simple,
  // stable, and accurate enough for moderate Reynolds numbers.
  if (rank_ != 0) return;
  const std::size_t N = nx_nbrs_ * ny_;
  const std::size_t x =1;
  for (std::size_t y = 0; y < ny_; ++y) {
    if (solid_[idx(x, y)]) continue;
    const double rho = 1.0;
    const double ux  = u_in_;
    const double uy  = 0.0;
    const double u2  = ux * ux + uy * uy;
    for (int i = 0; i < Q; ++i) {
      const double cu  = cx[i] * ux + cy[i] * uy;
      f_[i * N + idx(x, y)] = w[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    }
  }
}

void
LBM_MPI::apply_outlet()
{
  // Zero-gradient outlet: copy the second-to-last column into the last.
  if (rank_ != size_-1) return;
  if (nx_local_ < 2) return;
  const std::size_t N  = nx_nbrs_ * ny_;
  const std::size_t x  = nx_local_ ;
  const std::size_t xs = nx_local_ - 1;
  for (std::size_t y = 0; y < ny_; ++y) {
    for (int i = 0; i < Q; ++i) {
      f_[i * N + idx(x, y)] = f_[i * N + idx(xs, y)];
    }
  }
}

double
LBM_MPI::rho(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0;
  for (int i = 0; i < Q; ++i) r += f_[i * N + idx(x, y)];
  return r;
}

double
LBM_MPI::ux(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cx[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_MPI::uy(std::size_t x, std::size_t y) const
{
  const std::size_t N = nx_ * ny_;
  double r = 0.0, m = 0.0;
  for (int i = 0; i < Q; ++i) {
    const double fi = f_[i * N + idx(x, y)];
    r += fi;
    m += cy[i] * fi;
  }
  return (r > 0.0) ? m / r : 0.0;
}

double
LBM_MPI::vorticity(std::size_t x, std::size_t y) const
{
  if (x == 0 || x == nx_ - 1 || y == 0 || y == ny_ - 1) return 0.0;
  return 0.5 * ((uy(x + 1, y) - uy(x - 1, y)) - (ux(x, y + 1) - ux(x, y - 1)));
}

bool
LBM_MPI::is_solid(std::size_t x, std::size_t y) const
{
  return solid_[idx(x, y)] != 0;
}
