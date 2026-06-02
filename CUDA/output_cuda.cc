#include "output_mpi.hh"
#include "lbm_mpi.hh"

#include <hdf5.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>
#include <algorithm>

namespace {
std::size_t
local_nx_for_rank(std::size_t nx, int size, int rank)
{
  const std::size_t base = nx / std::size_t(size);
  const std::size_t rem  = nx % std::size_t(size);
  return base + (std::size_t(rank) < rem ? 1 : 0);
}

std::size_t
x_start_for_rank(std::size_t nx, int size, int rank)
{
  const std::size_t base = nx / std::size_t(size);
  const std::size_t rem  = nx % std::size_t(size);
  return std::size_t(rank) * base + std::min<std::size_t>(rem, std::size_t(rank));
}

std::vector<int>
make_counts(std::size_t nx, std::size_t ny, int size)
{
  std::vector<int> counts(size);
  for (int r = 0; r < size; ++r) {
    const std::size_t n = local_nx_for_rank(nx, size, r) * ny;
    counts[r] = int(n);  // fine unless your output array is enormous
  }
  return counts;
}

std::vector<int>
make_displs(const std::vector<int>& counts)
{
  std::vector<int> displs(counts.size(), 0);
  for (std::size_t r = 1; r < counts.size(); ++r) {
    displs[r] = displs[r - 1] + counts[r - 1];
  }
  return displs;
}

std::vector<double>
gather_double_field(const std::vector<double>& local,
                    std::size_t nx, std::size_t ny,
                    MPI_Comm comm, int rank, int size)
{
  const auto counts = make_counts(nx, ny, size);
  const auto displs = make_displs(counts);

  std::vector<double> gathered;
  if (rank == 0) {
    gathered.resize(nx * ny);
  }

  MPI_Gatherv(local.data(),
              int(local.size()),
              MPI_DOUBLE,
              rank == 0 ? gathered.data() : nullptr,
              counts.data(),
              displs.data(),
              MPI_DOUBLE,
              0,
              comm);

  return gathered;
}

std::vector<std::uint8_t>
gather_uint8_field(const std::vector<std::uint8_t>& local,
                   std::size_t nx, std::size_t ny,
                   MPI_Comm comm, int rank, int size)
{
  const auto counts = make_counts(nx, ny, size);
  const auto displs = make_displs(counts);

  std::vector<std::uint8_t> gathered;
  if (rank == 0) {
    gathered.resize(nx * ny);
  }

  MPI_Gatherv(local.data(),
              int(local.size()),
              MPI_UNSIGNED_CHAR,
              rank == 0 ? gathered.data() : nullptr,
              counts.data(),
              displs.data(),
              MPI_UNSIGNED_CHAR,
              0,
              comm);

  return gathered;
}

std::vector<double>
unpack_double_field(const std::vector<double>& gathered,
                    std::size_t nx, std::size_t ny,
                    int size)
{
  std::vector<double> global(nx * ny, 0.0);

  std::size_t offset = 0;
  for (int r = 0; r < size; ++r) {
    const std::size_t local_nx = local_nx_for_rank(nx, size, r);
    const std::size_t x_start  = x_start_for_rank(nx, size, r);

    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t lx = 0; lx < local_nx; ++lx) {
        global[y * nx + (x_start + lx)] =
          gathered[offset + y * local_nx + lx];
      }
    }

    offset += local_nx * ny;
  }

  return global;
}

std::vector<std::uint8_t>
unpack_uint8_field(const std::vector<std::uint8_t>& gathered,
                   std::size_t nx, std::size_t ny,
                   int size)
{
  std::vector<std::uint8_t> global(nx * ny, 0);

  std::size_t offset = 0;
  for (int r = 0; r < size; ++r) {
    const std::size_t local_nx = local_nx_for_rank(nx, size, r);
    const std::size_t x_start  = x_start_for_rank(nx, size, r);

    for (std::size_t y = 0; y < ny; ++y) {
      for (std::size_t lx = 0; lx < local_nx; ++lx) {
        global[y * nx + (x_start + lx)] =
          gathered[offset + y * local_nx + lx];
      }
    }

    offset += local_nx * ny;
  }

  return global;
}

std::vector<double>
compute_vorticity_global(const std::vector<double>& ux,
                         const std::vector<double>& uy,
                         std::size_t nx, std::size_t ny)
{
  std::vector<double> vor(nx * ny, 0.0);

  for (std::size_t y = 1; y + 1 < ny; ++y) {
    for (std::size_t x = 1; x + 1 < nx; ++x) {
      vor[y * nx + x] =
        0.5 * ((uy[y * nx + (x + 1)] - uy[y * nx + (x - 1)])
             - (ux[(y + 1) * nx + x] - ux[(y - 1) * nx + x]));
    }
  }

  return vor;
}


void
write_h5_double(const std::string & fname,
                const std::string & dset,
                const std::vector<double> & data,
                std::size_t nx, std::size_t ny,
                bool truncate)
{
  hid_t file = truncate
    ? H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT)
    : H5Fopen  (fname.c_str(), H5F_ACC_RDWR,                H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not open HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_IEEE_F64LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

void
write_h5_uint8(const std::string & fname,
               const std::string & dset,
               const std::vector<std::uint8_t> & data,
               std::size_t nx, std::size_t ny)
{
  hid_t file = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (file < 0) {
    std::cerr << "Could not create HDF5 file " << fname << "\n";
    return;
  }
  hsize_t dims[2] = { hsize_t(ny), hsize_t(nx) };
  hid_t   space   = H5Screate_simple(2, dims, nullptr);
  hid_t   dataset = H5Dcreate2(file, dset.c_str(), H5T_STD_U8LE,
                               space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  H5Dclose(dataset);
  H5Sclose(space);
  H5Fclose(file);
}

std::string
strip_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? p : p.substr(pos + 1);
}

std::string
parent_dir(const std::string & p)
{
  const auto pos = p.find_last_of("/\\");
  return (pos == std::string::npos) ? std::string{} : p.substr(0, pos);
}

// Recursive `mkdir -p`. Errors other than "already exists" are silently
// ignored so that the file-creation step downstream surfaces the real
// problem to the user.
void
mkdir_p(const std::string & path)
{
  if (path.empty()) return;
  std::string acc;
  for (char c : path) {
    if ((c == '/' || c == '\\') && !acc.empty()) {
      ::mkdir(acc.c_str(), 0755);
    }
    acc.push_back(c);
  }
  if (!acc.empty()) ::mkdir(acc.c_str(), 0755);
}

std::string
indexed_suffix(std::size_t i)
{
  std::ostringstream os;
  os << std::setw(6) << std::setfill('0') << i;
  return os.str();
}

}  // namespace

XDMFWriter_MPI::XDMFWriter_MPI(const std::string & prefix, std::size_t nx, std::size_t ny, MPI_Comm comm)
  : prefix_(prefix), basename_(strip_dir(prefix)),
    nx_(nx), ny_(ny), has_mask_(false), comm_(comm)
{
  MPI_Comm_rank(comm_, &rank_);
  MPI_Comm_size(comm_, &size_);
  if (rank_ == 0) {
    mkdir_p(parent_dir(prefix_));
  }
}

void
XDMFWriter_MPI::write_mask(const LBM_MPI & solver)
{
  const std::size_t local_nx = solver.nx_local();

  std::vector<std::uint8_t> mask_local(local_nx * ny_);

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t lx = 0; lx < local_nx; ++lx) {
      const std::size_t local_x = lx + 1;  // skip left ghost column
      mask_local[y * local_nx + lx] =
        solver.is_solid(local_x, y) ? 1 : 0;
    }
  }

  std::vector<std::uint8_t> gathered =
    gather_uint8_field(mask_local, nx_, ny_, comm_, rank_, size_);

  if (rank_ == 0) {
    std::vector<std::uint8_t> mask =
      unpack_uint8_field(gathered, nx_, ny_, size_);

    write_mask_array(mask);
  }
}

void
XDMFWriter_MPI::write_mask_array(const std::vector<std::uint8_t>& mask)
{
  if (rank_ != 0) return;

  if (mask.size() != nx_ * ny_) {
    std::cerr << "Faulty mask size!";
    return;
  }

  write_h5_uint8(prefix_ + "_mask.h5", "solid", mask, nx_, ny_);
  has_mask_ = true;
  write_root_xdmf();
}

void
XDMFWriter_MPI::write_snapshot(const LBM_MPI & solver, double t)
{
  const std::size_t local_nx = solver.nx_local();

  std::vector<double> rho_local(local_nx * ny_);
  std::vector<double> ux_local (local_nx * ny_);
  std::vector<double> uy_local (local_nx * ny_);

  for (std::size_t y = 0; y < ny_; ++y) {
    for (std::size_t lx = 0; lx < local_nx; ++lx) {
      const std::size_t local_x = lx + 1;  // real cell, not ghost
      const std::size_t k = y * local_nx + lx;

      rho_local[k] = solver.rho(local_x, y);
      ux_local [k] = solver.ux (local_x, y);
      uy_local [k] = solver.uy (local_x, y);
    }
  }

  std::vector<double> rho_gathered =
    gather_double_field(rho_local, nx_, ny_, comm_, rank_, size_);

  std::vector<double> ux_gathered =
    gather_double_field(ux_local, nx_, ny_, comm_, rank_, size_);

  std::vector<double> uy_gathered =
    gather_double_field(uy_local, nx_, ny_, comm_, rank_, size_);

  if (rank_ == 0) {
    std::vector<double> rho =
      unpack_double_field(rho_gathered, nx_, ny_, size_);

    std::vector<double> ux =
      unpack_double_field(ux_gathered, nx_, ny_, size_);

    std::vector<double> uy =
      unpack_double_field(uy_gathered, nx_, ny_, size_);

    std::vector<double> vor =
      compute_vorticity_global(ux, uy, nx_, ny_);

    write_snapshot_arrays(rho, ux, uy, vor, t);
  }
}
void
XDMFWriter_MPI::write_snapshot_arrays(const std::vector<double> & rho,
                                  const std::vector<double> & ux,
                                  const std::vector<double> & uy,
                                  const std::vector<double> & vor,
                                  double t)
{
  if (rank_ != 0) return;
  if (rho.size() != nx_ * ny_ || ux.size() != nx_ * ny_ || uy.size() != nx_ * ny_ || vor.size() != nx_ * ny_)
  {
    std::cerr << "Faulty field sizes!";
    return;
  }
  const std::size_t step_idx = times_.size();
  const std::string fname    = prefix_ + "_" + indexed_suffix(step_idx) + ".h5";

  write_h5_double(fname, "rho",       rho, nx_, ny_, /*truncate=*/true);
  write_h5_double(fname, "ux",        ux,  nx_, ny_, /*truncate=*/false);
  write_h5_double(fname, "uy",        uy,  nx_, ny_, /*truncate=*/false);
  write_h5_double(fname, "vorticity", vor, nx_, ny_, /*truncate=*/false);

  times_.push_back(t);
  write_root_xdmf();
}

void
XDMFWriter_MPI::write_root_xdmf() const
{
  const std::string xmf_path = prefix_ + ".xdmf";
  std::ofstream xmf(xmf_path);
  if (!xmf) {
    std::cerr << "Could not open " << xmf_path << "\n";
    return;
  }

  xmf << "<?xml version=\"1.0\" ?>\n"
      << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n"
      << "<Xdmf Version=\"3.0\">\n"
      << "  <Domain>\n"
      << "    <Grid Name=\"LBM\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";

  for (std::size_t i = 0; i < times_.size(); ++i) {
    const std::string tag = indexed_suffix(i);
    const std::string h5  = basename_ + "_" + tag + ".h5";

    xmf << "      <Grid Name=\"step_" << tag << "\" GridType=\"Uniform\">\n"
        << "        <Time Value=\"" << times_[i] << "\"/>\n"
        << "        <Topology TopologyType=\"2DCoRectMesh\" Dimensions=\""
        << ny_ << " " << nx_ << "\"/>\n"
        << "        <Geometry GeometryType=\"ORIGIN_DXDY\">\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">0.0 0.0</DataItem>\n"
        << "          <DataItem Format=\"XML\" Dimensions=\"2\">1.0 1.0</DataItem>\n"
        << "        </Geometry>\n";

    auto write_attr = [&](const char * name, const char * dset) {
      xmf << "        <Attribute Name=\"" << name
          << "\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"Float\" Precision=\"8\">"
          << h5 << ":/" << dset
          << "</DataItem>\n"
          << "        </Attribute>\n";
    };
    write_attr("rho",       "rho");
    write_attr("ux",        "ux");
    write_attr("uy",        "uy");
    write_attr("vorticity", "vorticity");

    if (has_mask_) {
      xmf << "        <Attribute Name=\"solid\" AttributeType=\"Scalar\" Center=\"Node\">\n"
          << "          <DataItem Format=\"HDF\" Dimensions=\""
          << ny_ << " " << nx_ << "\" NumberType=\"UChar\">"
          << basename_ << "_mask.h5:/solid"
          << "</DataItem>\n"
          << "        </Attribute>\n";
    }

    xmf << "      </Grid>\n";
  }

  xmf << "    </Grid>\n"
      << "  </Domain>\n"
      << "</Xdmf>\n";
}
