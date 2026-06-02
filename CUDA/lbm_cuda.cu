#include "lbm_cuda.hh"
#include <algorithm>

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
const int LBM_CUDA::cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
const int LBM_CUDA::cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

const double LBM_CUDA::w[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

const int LBM_CUDA::opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

LBM_CUDA::LBM_CUDA(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r)
: nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0), block_x_(16), block_y_(16)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  cudaMalloc(&f_d_,     9 * nx_ *ny_* sizeof(double));
  cudaMalloc(&ftmp_d_,  9 * nx_*ny_ * sizeof(double));
  cudaMalloc(&solid_d_, nx_*ny_ * sizeof(std::uint8_t));

  cudaMalloc(&rho_d_,   nx_*ny_ * sizeof(double));
  cudaMalloc(&ux_d_,    nx_*ny_ * sizeof(double));
  cudaMalloc(&d_uy_,    nx_*ny_ * sizeof(double));
  cudaMalloc(&vort_d_,  nx_*ny_ * sizeof(double));

  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);


  // No-slip top and bottom walls.
  clear_solid_kernel<<<grid, block>>>(solid_d_, nx_, ny_);

  dim3 block_x(256);
  dim3 grid_x((nx_ + block_x.x - 1) / block_x.x);
  mark_walls_kernel<<<grid_x, block_x>>>(solid_d_, nx_, ny_);

  mark_obstacle(cyl_x, cyl_y, cyl_r);
}

LBM_CUDA::~LBM_CUDA()
{
  cudaFree(f_d_);
  cudaFree(ftmp_d_);
  cudaFree(solid_d_);
  cudaFree(rho_d_);
  cudaFree(ux_d_);
  cudaFree(uy_d_);
  cudaFree(vort_d_);
}

void
LBM_MPI::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM_MPI::mark_obstacle(double c_x, double c_y, double r)
{
  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);
  mark_obstacle_kernel<<<grid, block>>>(solid_d_, nx_, ny_, c_x, c_y, r);
}

void
LBM_MPI::initialize()
{
  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);
  init_kernel<<<grid, block>>>(f_d_, solid_d_, nx_, ny_, u_in_);
}

void
LBM_CUDA::step()
{
  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

  collide_kernel<<<grid, block>>>(f_d_, solid_d_, nx_, ny_, tau_);
  bounce_back_kernel<<<grid, block>>>(f_d_, solid_d_, nx_, ny_);
  stream_kernel<<<grid, block>>>(f_d_, ftmp_d_, nx_, ny_);

  dim3 block_y_1d(256);

  apply_inlet_kernel<<<grid_y, block_y_1d>>>(f_d_, solid_d_, nx_, ny_, u_in_);
  apply_outlet_kernel<<<grid_y, block_y_1d>>>(f_d_, nx_, ny_);
}

void
LBM_CUDA::copy_mask_to_host(std::vector<std::uint8_t>& mask) const
{
  mask.resize(nx_ * ny_);
  cudaMemcpy(mask.data(), solid_d_, nx_ * ny_ * sizeof(std::uint8_t), cudaMemcpyDeviceToHost);
}


void
LBM_CUDA::copy_vars_to_host(std::vector<double>& rho, std::vector<double>& ux, std::vector<double>& uy, std::vector<double>& vorticity) const
{
  rho.resize(nx_*ny_);
  ux.resize(nx_*ny_);
  uy.resize(mx_*ny_);
  vorticity.resize(nx_*ny_);

  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

  macroscopic_kernel<<<grid, block>>>(f_d_, rho_d_, ux_d_, uy_d_, nx_, ny_);

  vorticity_kernel<<<grid, block>>>(ux_d_, uy_d_, vort_d_, nx_, ny_);

  cudaMemcpy(rho.data(), rho_d_, nx_*ny_ * sizeof(double), cudaMemcpyDeviceToHost));
  cudaMemcpy(ux.data(), ux_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost));
  cudaMemcpy(uy.data(), uy_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost));
  cudaMemcpy(vorticity.data(), vort_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost));
}