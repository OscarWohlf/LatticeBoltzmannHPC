#include "lbm_cuda.hh"
#include <algorithm>
#include <cuda_runtime.h>
#include <cstdint>

// D2Q9 lattice constants. Indexing convention used throughout:
//   0: rest         5: NE
//   1: E            6: NW
//   2: N            7: SW
//   3: W            8: SE
//   4: S
__constant__ int cx_d[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
__constant__ int cy_d[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

__constant__ double w_d[9] = {
  4.0 / 9.0,
  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,  1.0 / 9.0,
  1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0
};

__constant__ int opp_d[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};

__device__ inline std::size_t
idx(std::size_t x, std::size_t y, std::size_t nx)
{
  return y * nx + x;
}

__device__ inline std::size_t
fidx(int i, std::size_t x, std::size_t y, std::size_t nx, std::size_t ny)
{
  return std::size_t(i) * nx * ny + idx(x, y, nx);
}


__global__
void clear_solid_kernel(std::uint8_t* solid, std::size_t nx, std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  solid[idx(x, y, nx)] = 0;
}

__global__
void mark_walls_kernel(std::uint8_t* solid, std::size_t nx,std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= nx) return;
  solid[idx(x, 0,      nx)] = 1;
  solid[idx(x, ny - 1, nx)] = 1;
}

__global__
void mark_obstacle_kernel(std::uint8_t* solid, std::size_t nx,  std::size_t ny,  double c_x,  double c_y,  double r)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const double dx = double(x) - c_x;
  const double dy = double(y) - c_y;
  if (dx*dx+ dy* dy <= r * r) {
    solid[idx(x, y, nx)] = 1;
  }
}

__global__
void init_kernel(double* f,  const std::uint8_t* solid, std::size_t nx,std::size_t ny,double u_in)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t k = idx(x, y, nx);
  const std::size_t N = nx * ny;

  const double ux  = solid[k] ? 0.0 : u_in;
  const double uy  = 0.0;

  const double u2  = ux * ux + uy * uy;

  const double rho = 1.0;

  for (int i = 0; i < 9; ++i) {
    const double cu = cx_d[i] * ux + cy_d[i] * uy;
    const double feq =w_d[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f[i * N + k] = feq;
  }
}

__global__
void collide_kernel(double* f,const std::uint8_t* solid, std::size_t nx,  std::size_t ny, double tau)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;

  const std::size_t N = nx * ny;
  const std::size_t k = idx(x, y, nx);

  double rho = 0.0;
  double mx  = 0.0;
  double my  = 0.0;

  for (int i = 0; i < 9; ++i) {
    const double fi = f[i * N + k];
    rho += fi;
    mx  += cx_d[i] * fi;
    my  += cy_d[i] * fi;
  }

  const double ux = (rho > 0.0) ? mx / rho : 0.0;
  const double uy = (rho > 0.0) ? my / rho : 0.0;
  const double u2 = ux * ux + uy * uy;

  for (int i = 0; i < 9; ++i) {
    const double cu = cx_d[i] * ux + cy_d[i] * uy;
    const double feq =  w_d[i] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * u2);
    f[i * N + k] += -(1.0 / tau) * (f[i * N + k] - feq);
  }
}

__global__
void bounce_back_kernel(double* f,   const std::uint8_t* solid,  std::size_t nx,  std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t k = idx(x, y, nx);
  if (!solid[k]) return;

  const std::size_t N = nx * ny;

  for (int i = 1; i < 9; ++i) {
    const int j = opp_d[i];
    if (i < j) {
      const double tmp = f[i * N + k];
      f[i * N + k] = f[j * N + k];
      f[j * N + k] = tmp;
    }
  }
}

__global__
void stream_kernel(const double* f,  double* ftmp, std::size_t nx, std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t N = nx * ny;
  const std::size_t k = idx(x, y, nx);

  for (int i = 0; i < 9; ++i) {
    const long sx = long(x) - cx_d[i];
    const long sy = long(y) - cy_d[i];

    if (sx >= 0 && sx < long(nx) && sy >= 0 && sy < long(ny)) {
      ftmp[i * N + k] = f[i * N + idx(std::size_t(sx), std::size_t(sy), nx)];
    } else {
      ftmp[i * N + k] = f[i * N + k];
    }
  }
}

__global__
void apply_inlet_kernel(double* f,  const std::uint8_t* solid, std::size_t nx,  std::size_t ny,  double u_in)
{
  const std::size_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= ny) return;
  const std::size_t x = 0;
  const std::size_t k = idx(x, y, nx);
  const std::size_t N = nx * ny;

  if (solid[k]) return;

  const double rho = 1.0;
  const double ux  = u_in;
  const double uy  = 0.0;
  const double u2  = ux * ux + uy * uy;

  for (int i = 0; i < 9; ++i) {
    const double cu = cx_d[i] * ux + cy_d[i] * uy;
    f[i*N +k] =  w_d[i] * rho * (1.0 + 3.0* cu + 4.5 * cu * cu - 1.5 * u2);
  }
}

__global__
void apply_outlet_kernel(double* f, std::size_t nx,  std::size_t ny)
{
  const std::size_t y = blockIdx.x* blockDim.x + threadIdx.x;

  if (y >= ny || nx < 2) return;
  const std::size_t x  = nx - 1;
  const std::size_t xs = nx - 2;

  for (int i = 0; i < 9; ++i) {
    f[fidx(i, x,  y, nx, ny)] = f[fidx(i, xs, y, nx, ny)];
  }
}


__global__
void macro_updates_kernel(const double* f, double* rho,  double* ux,  double* uy,  std::size_t nx,std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= nx || y >= ny) return;
  const std::size_t k = idx(x, y, nx);
  const std::size_t N = nx * ny;

  double r  = 0.0;
  double mx = 0.0;
  double my = 0.0;

  for (int i = 0; i < 9; ++i) {
    const double fi = f[i * N + k];
    r  += fi;
    mx += cx_d[i] * fi;
    my += cy_d[i] * fi;
  }
  rho[k] = r;
  ux[k]  = (r > 0.0) ? mx / r : 0.0;
  uy[k]  = (r > 0.0) ? my / r : 0.0;
}


__global__
void vort_kernel(const double* ux,  const double* uy,  double* vort,  std::size_t nx,  std::size_t ny)
{
  const std::size_t x = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= nx || y >= ny) return;
  const std::size_t k = idx(x, y, nx);

  if (x == 0 || x == nx - 1 || y == 0 || y == ny - 1) {
    vort[k] = 0.0;
    return;
  }

  vort[k] =
    0.5 * ((uy[idx(x + 1, y, nx)] - uy[idx(x - 1, y, nx)])
         - (ux[idx(x, y + 1, nx)] - ux[idx(x, y - 1, nx)]));
}


LBM_CUDA::LBM_CUDA(std::size_t nx, std::size_t ny,
         double u_in, double Re,
         double cyl_x, double cyl_y, double cyl_r, int block_x, int block_y)
: nx_(nx), ny_(ny), u_in_(u_in), tau_(0.0), block_x_(block_x), block_y_(block_y)
{
  // ν = c_s^2 (τ - 1/2) with c_s^2 = 1/3, and Re = u_in * D / ν.
  const double nu = u_in_ * (2.0 * cyl_r) / Re;
  tau_ = 3.0 * nu + 0.5;

  cudaMalloc(&f_d_,     9 * nx_ *ny_* sizeof(double));
  cudaMalloc(&ftmp_d_,  9 * nx_*ny_ * sizeof(double));
  cudaMalloc(&solid_d_, nx_*ny_ * sizeof(std::uint8_t));

  cudaMalloc(&rho_d_,   nx_*ny_ * sizeof(double));
  cudaMalloc(&ux_d_,    nx_*ny_ * sizeof(double));
  cudaMalloc(&uy_d_,    nx_*ny_ * sizeof(double));
  cudaMalloc(&vort_d_,  nx_*ny_ * sizeof(double));

  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);


  // No-slip top and bottom walls.
  clear_solid_kernel<<<grid, block>>>(solid_d_, nx_, ny_);

  dim3 block_wall_x(256);
  dim3 grid_wall_x((nx_ + block_wall_x.x - 1) / block_wall_x.x);
  mark_walls_kernel<<<grid_wall_x, block_wall_x>>>(solid_d_, nx_, ny_);

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
LBM_CUDA::add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r)
{
  if (cyl2_r > 0.0) mark_obstacle(cyl2_x, cyl2_y, cyl2_r);
}

void
LBM_CUDA::mark_obstacle(double c_x, double c_y, double r)
{
  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);
  mark_obstacle_kernel<<<grid, block>>>(solid_d_, nx_, ny_, c_x, c_y, r);
}

void
LBM_CUDA::initialize()
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
  std::swap(f_d_, ftmp_d_);
  dim3 block_y_1d(256);
  dim3 grid_y((ny_ + block_y_1d.x - 1) / block_y_1d.x);
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
  uy.resize(nx_*ny_);
  vorticity.resize(nx_*ny_);

  dim3 block(block_x_, block_y_);
  dim3 grid((nx_ + block.x - 1) / block.x, (ny_ + block.y - 1) / block.y);

  macro_updates_kernel<<<grid, block>>>(f_d_, rho_d_, ux_d_, uy_d_, nx_, ny_);

  vort_kernel<<<grid, block>>>(ux_d_, uy_d_, vort_d_, nx_, ny_);

  cudaMemcpy(rho.data(), rho_d_, nx_*ny_ * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(ux.data(), ux_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(uy.data(), uy_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost);
  cudaMemcpy(vorticity.data(), vort_d_, nx_ * ny_ * sizeof(double), cudaMemcpyDeviceToHost);
}

void
LBM_CUDA::copy_probe_to_host(std::size_t x,std::size_t y, double& ux, double& uy) const
{
  if (x >= nx_ || y >= ny_) {
    ux = 0.0;
    uy = 0.0;
    return;
  }
  const std::size_t N = nx_ * ny_;
  const std::size_t k = y * nx_ + x;

  double f_host[9];
  for (int i = 0; i < 9; ++i) {
    cudaMemcpy(&f_host[i], f_d_ + i * N + k, sizeof(double),cudaMemcpyDeviceToHost);
  }

  const int cx[9] = { 0,  1,  0, -1,  0,  1, -1, -1,  1};
  const int cy[9] = { 0,  0,  1,  0, -1,  1,  1, -1, -1};

  double mx  = 0.0;
  double my  = 0.0;
  double rho = 0.0;

  for (int i = 0; i < 9; ++i) {
    rho += f_host[i];
    mx  += cx[i] * f_host[i];
    my  += cy[i] * f_host[i];
  }
  ux = (rho > 0.0) ? mx / rho : 0.0;
  uy = (rho > 0.0) ? my / rho : 0.0;
}