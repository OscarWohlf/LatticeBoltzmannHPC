#include "lbm_mpi.hh"
#include "output_mpi.hh"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include "mpi.h"

namespace {

using Args = std::unordered_map<std::string, std::string>;

Args
parse_args(int argc, char ** argv)
{
  Args kv;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto eq = a.find('=');
    if (eq == std::string::npos) {
      std::cerr << "Bad argument '" << a << "' (expected key=value)\n";
      std::exit(1);
    }
    kv[a.substr(0, eq)] = a.substr(eq + 1);
  }
  return kv;
}

template <typename T>
T
get(const Args & kv, const std::string & key, T def)
{
  auto it = kv.find(key);
  if (it == kv.end()) return def;
  std::istringstream iss(it->second);
  T v; iss >> v;
  return v;
}

std::string
get_string(const Args & kv, const std::string & key, const std::string & def)
{
  auto it = kv.find(key);
  return (it == kv.end()) ? def : it->second;
}

}  // namespace

int
main(int argc, char ** argv)
{
  const Args kv = parse_args(argc, argv);
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Grid + physics.
  const std::size_t nx    = get<std::size_t>(kv, "nx",    800);
  const std::size_t ny    = get<std::size_t>(kv, "ny",    400);
  const double      Re    = get<double>     (kv, "re",    100.0);
  const double      u_in  = get<double>     (kv, "u_in",  0.05);
  const std::size_t steps = get<std::size_t>(kv, "steps", 60000);

  // Cylinder geometry. Defaults: at (nx/4, ny/2) with radius ny/40
  // (i.e. cylinder diameter = ny/20, ~5% blockage). At Re = 100 this
  // setup reproduces the classical Strouhal number St ~ 0.16.
  const double cx0 = get<double>(kv, "cyl_x", double(nx) * 0.25);
  const double cy0 = get<double>(kv, "cyl_y", double(ny) * 0.50);
  const double cr0 = get<double>(kv, "cyl_r", double(ny) * 0.025);

  // Optional second cylinder. Disabled if cyl2_r <= 0.
  const double cx1 = get<double>(kv, "cyl2_x", -1.0);
  const double cy1 = get<double>(kv, "cyl2_y", -1.0);
  const double cr1 = get<double>(kv, "cyl2_r", -1.0);

  // Output.
  const std::size_t every     = get<std::size_t>(kv, "every", 500);
  const std::string out_pref  = get_string(kv, "out", "out/lbm");
  const std::string probe_csv = get_string(kv, "probe", "probe.csv");

  // Probe location: about 4 diameters downstream, on the cylinder centerline.
  const std::size_t px = get<std::size_t>(kv, "probe_x",
                                          std::size_t(cx0 + 8.0 * cr0));
  const std::size_t py = get<std::size_t>(kv, "probe_y", std::size_t(cy0));


  LBM_MPI solver(nx, ny, u_in, Re, cx0, cy0, cr0, MPI_COMM_WORLD);
  if (cr1 > 0.0) solver.add_second_cylinder(cx1, cy1, cr1);
  solver.initialize();
  if (rank == 0) {
    std::cout << "LBM 2D D2Q9 BGK\n"
              << "  grid          : " << nx << " x " << ny << "\n"
              << "  Re            : " << Re << "\n"
              << "  u_in          : " << u_in << "\n"
              << "  tau           : " << solver.tau() << "\n"
              << "  cylinder      : (" << cx0 << ", " << cy0
              << "), r = " << cr0 << "\n";

    if (cr1 > 0.0) {
        std::cout << "  cylinder #2   : (" << cx1 << ", " << cy1
              << "), r = " << cr1 << "\n";
      }
  std::cout << "  steps         : " << steps << "\n"
            << "  output every  : " << every << " (0 = off)\n"
            << "  output prefix : " << out_pref << "\n"
            << "  probe at      : (" << px << ", " << py << ")\n"
            << "  probe csv     : " << probe_csv << "\n";
  }
  std::ofstream probe;

  bool owns_probe = solver.owns_global_x(px);
  std::size_t probe_local_x = owns_probe ? solver.get_local_x(px) : 0;

  if (owns_probe) {
    probe.open(probe_csv);
    probe << "step,ux,uy\n";
   }
  XDMFWriter_MPI writer(out_pref, nx, ny, MPI_COMM_WORLD);
  if (every > 0) writer.write_mask(writer,solver);
  MPI_Barrier(MPI_COMM_WORLD);
  using clk = std::chrono::high_resolution_clock;
  const auto t0 = clk::now();

  for (std::size_t step = 1; step <= steps; ++step) {
    solver.step();
    if (owns_probe) {
      probe << step << ',' << solver.ux(probe_local_x, py) << ',' << solver.uy(probe_local_x, py) << '\n';
    }
    if (every > 0 && step % every == 0) {
      writer.write_snapshot(solver, writer, double(step));
      if (rank == 0) {
        std::cout << "\r  step " << step << " / " << steps << std::flush;
      }
    }
  }
  if (rank == 0 && every > 0) std::cout << "\n";
  MPI_Barrier(MPI_COMM_WORLD); 
  const double dt    = std::chrono::duration<double>(clk::now() - t0).count();
  if (rank == 0) {
    const double mlups = double(nx) * double(ny) * double(steps) / dt / 1.0e6;
    std::cout << "Wall time : " << dt    << " s\n"
              << "MLUPS     : " << mlups << "\n";
  }
  MPI_Finalize();
  return 0;
}
