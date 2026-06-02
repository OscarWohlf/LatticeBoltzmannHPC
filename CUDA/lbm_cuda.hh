#ifndef LBM_CUDA_HH
#define LBM_CUDA_HH

#include <cstddef>
#include <cstdint>
#include <vector>

/**
*
 * @brief 2D Lattice Boltzmann solver, D2Q9 lattice with BGK collision.
 *
 * The solver simulates incompressible flow past one (or two) circular
 * cylinders inside a rectangular channel with no-slip top and bottom walls.
 * The inlet (x = 0) prescribes a uniform horizontal velocity; the outlet
 * (x = nx - 1) is a simple zero-gradient copy from the column to its left.
 *
 * Distributions are stored in structure-of-arrays layout:
 *   f_[ i * (nx*ny) + y*nx + x ]   for direction i in [0, 9).
 *
 * Lattice units are used throughout (dx = dt = 1, c = 1, c_s^2 = 1/3).
 */
class LBM_CUDA
{
public:
  static constexpr int Q = 9;

  /**
   * @param nx       Number of cells along x.
   * @param ny       Number of cells along y.
   * @param u_in     Inlet velocity in lattice units (must be << 1/sqrt(3)).
   * @param Re       Target Reynolds number, based on cylinder diameter.
   * @param cyl_x    Center of the (first) cylinder along x, in cell units.
   * @param cyl_y    Center of the (first) cylinder along y, in cell units.
   * @param cyl_r    Radius of the (first) cylinder, in cell units.
   */
  LBM_CUDA(std::size_t nx, std::size_t ny,
      double u_in, double Re,
      double cyl_x, double cyl_y, double cyl_r,
      int block_x = 16, int block_y = 16);

  ~LBM_CUDA();

  /// Add a second circular obstacle. No-op if r2 <= 0.
  void add_second_cylinder(double cyl2_x, double cyl2_y, double cyl2_r);

  /// Set f to the equilibrium distribution with rho = 1, u = (u_in, 0)
  /// on every fluid cell, and (0, 0) on solid cells.
  void initialize();

  /// Advance the simulation by one time step.
  void step();

  void copy_mask_to_host(std::vector<std::uint8_t>& mask) const;

  void copy_vars_to_host(std::vector<double>& rho, std::vector<double>& ux, std::vector<double>& uy, std::vector<double>& vorticity) const;

  std::size_t nx() const { return nx_; }
  std::size_t ny() const { return ny_; }
  double tau() const { return tau_; }

  double u_in() const { return u_in_; }

  void copy_probe_to_host(std::size_t x,  std::size_t y, double& ux, double& uy) const;

private:
  void mark_obstacle (double c_x, double c_y, double r);
  std::size_t nx_, ny_;
  double u_in_;
  double tau_;

  int block_x_;
  int block_y_;

  double* rho_d_ = nullptr;
  double* ux_d_ = nullptr;
  double* uy_d_ = nullptr;
  double* vort_d_ = nullptr;

  double* f_d_ = nullptr;     ///< Current distributions, size 9*nx*ny.
  double* ftmp_d_ = nullptr;   ///< Scratch buffer for streaming.
  std::uint8_t* solid_d_ = nullptr;  ///< 0 = fluid, 1 = solid.  Size nx*ny.
};

#endif  // LBM_CUDA_HH
