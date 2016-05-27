#include "two_layer_net.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define MPI_SAFE_CALL( call ) do {                               \
    int err = call;                                              \
    if (err != MPI_SUCCESS) {                                    \
        fprintf(stderr, "MPI error %d in file '%s' at line %i",  \
               err, __FILE__, __LINE__);                         \
        exit(1);                                                 \
    } } while(0)

double norms (TwoLayerNet &nn) {
      double norm_sum = 0;

      for (int i = 0; i < nn.num_layers; ++i)  {
        norm_sum += arma::accu (arma::square (nn.W[i]));
      }

      return norm_sum;
}

void write_cpudata_tofile(TwoLayerNet &nn, int iter)
{
  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  nn.W[0].save(s.str(), arma::raw_ascii);
  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  nn.W[1].save(t.str(), arma::raw_ascii);
  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  nn.b[0].save(u.str(), arma::raw_ascii);
  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  nn.b[1].save(v.str(), arma::raw_ascii);  
}

void write_diff_gpu_cpu(TwoLayerNet &nn, int iter, std::ofstream& error_file)
{
  arma::mat A, B, C, D; 

  std::stringstream s;
  s << "Outputs/CPUmats/SequentialW0-" << iter << ".mat";
  A.load(s.str(), arma::raw_ascii);
  double max_errW0 = arma::norm(nn.W[0]-A, "inf")/arma::norm(A, "inf");
  double L2_errW0  = arma::norm(nn.W[0]-A,2)/arma::norm(A,2);

  std::stringstream t;
  t << "Outputs/CPUmats/SequentialW1-" << iter << ".mat";
  B.load(t.str(), arma::raw_ascii);
  double max_errW1 = arma::norm(nn.W[1]-B, "inf")/arma::norm(B, "inf");
  double L2_errW1  = arma::norm(nn.W[1]-B,2)/arma::norm(B,2);

  std::stringstream u;
  u << "Outputs/CPUmats/Sequentialb0-" << iter << ".mat";
  C.load(u.str(), arma::raw_ascii);
  double max_errb0 = arma::norm(nn.b[0]-C, "inf")/arma::norm(C, "inf");
  double L2_errb0  = arma::norm(nn.b[0]-C,2)/arma::norm(C,2);

  std::stringstream v;
  v << "Outputs/CPUmats/Sequentialb1-" << iter << ".mat";
  D.load(v.str(), arma::raw_ascii);
  double max_errb1 = arma::norm(nn.b[1]-D, "inf")/arma::norm(D, "inf");
  double L2_errb1  = arma::norm(nn.b[1]-D,2)/arma::norm(D,2);

  int ow = 15;
  if( iter == 0 ) {
    error_file << std::left<< std::setw(ow) << "Iteration" << std::left<< std::setw(ow) << "Max Err W0" << std::left << std::setw(ow) << "Max Err W1" 
    << std::left<< std::setw(ow) << "Max Err b0" << std::left<< std::setw(ow) << "Max Err b1" << std::left << std::setw(ow) << "L2 Err W0" << std::left 
    << std::setw(ow) << "L2 Err W1" << std::left<< std::setw(ow) << "L2 Err b0" << std::left<< std::setw(ow) << "L2 Err b1" << "\n";
  }
  error_file << std::left << std::setw(ow) << iter << std::left << std::setw(ow) << max_errW0 << std::left << std::setw(ow) << max_errW1 << 
  std::left << std::setw(ow) << max_errb0 << std::left << std::setw(ow) << max_errb1 << std::left<< std::setw(ow) << L2_errW0 << std::left << 
  std::setw(ow) << L2_errW1 << std::left << std::setw(ow) << L2_errb0 << std::left<< std::setw(ow) << L2_errb1 << "\n";
  
} 


void feedforward (TwoLayerNet &nn, const arma::mat& X, struct cache& cache)
{
  cache.z.resize(2);
  cache.a.resize(2);

  // std::cout << "W[0].n_rows " << W[0].n_rows << "\n";
  assert (X.n_cols == nn.W[0].n_cols);
  cache.X = X;
  int N = X.n_rows;

  arma::mat z1 = X * nn.W[0].t() + arma::repmat(nn.b[0], N, 1);
  cache.z[0] = z1;

  // std::cout << "Computing a1 " << "\n";
  arma::mat a1;
  sigmoid (z1, a1);
  cache.a[0] = a1;

  // std::cout << "Computing z2 " << "\n";
  assert (a1.n_cols == nn.W[1].n_cols);
  arma::mat z2 = a1 * nn.W[1].t() + arma::repmat(nn.b[1], N, 1);
  cache.z[1] = z2;

  // std::cout << "Computing a2 " << "\n";
  arma::mat a2;
  softmax (z2, a2);
  cache.a[1] = cache.yc = a2;

/*
  std::cout << "X " << arma::size(X) << std::endl;
  std::cout << "z1 " << arma::size(z1) << std::endl;
  std::cout << "z2 " << arma::size(z2) << std::endl;
  std::cout << "a1 " << arma::size(a1) << std::endl;
  std::cout << "a2 " << arma::size(a2) << std::endl;
  std::cout << "nn.W[0] " << arma::size(nn.W[0]) << std::endl;
  std::cout << "nn.W[1] " << arma::size(nn.W[1]) << std::endl;
  std::cout << "nn.b[0] " << arma::size(nn.b[0]) << std::endl;
  std::cout << "nn.b[1] " << arma::size(nn.b[1]) << std::endl;
*/

}

/*
 * Computes the gradients of the cost w.r.t each param.
 * MUST be called after feedforward since it uses the bpcache.
 * @params y : N x C one-hot row vectors
 * @params bpcache : Output of feedforward.
 * @params bpgrads: Returns the gradients for each param
 */
void backprop (TwoLayerNet &nn, const arma::mat& y, double reg, const struct cache& bpcache, struct grads& bpgrads)
{
  bpgrads.dW.resize(2);
  bpgrads.db.resize(2);
  int N = y.n_rows;

  // std::cout << "backprop " << bpcache.yc << "\n";
  arma::mat diff = (1.0 / N) * (bpcache.yc - y);
  bpgrads.dW[1] = diff.t() * bpcache.a[0] + reg * nn.W[1];
  bpgrads.db[1] = arma::sum (diff, 0);
  arma::mat da1 = diff * nn.W[1];

  arma::mat dz1 = da1 % bpcache.a[0] % (1 - bpcache.a[0]);

  bpgrads.dW[0] = dz1.t() * bpcache.X + reg * nn.W[0];
  bpgrads.db[0] = arma::sum(dz1, 0);

/*
  std::cout << "y " << arma::size(y) << std::endl;
  std::cout << "da1 " << arma::size(da1) << std::endl;
  std::cout << "dz1 " << arma::size(dz1) << std::endl;
  std::cout << "bpgrads.dW[0] " << arma::size(bpgrads.dW[0]) << std::endl;
  std::cout << "bpgrads.dW[1] " << arma::size(bpgrads.dW[1]) << std::endl;
  std::cout << "bpgrads.db[0] " << arma::size(bpgrads.db[0]) << std::endl;
  std::cout << "bpgrads.db[1] " << arma::size(bpgrads.db[1]) << std::endl;
  std::cout << "diff " << arma::size(diff) << std::endl;
*/

}

/*
 * Computes the Cross-Entropy loss function for the neural network.
 */
double loss (TwoLayerNet &nn, const arma::mat& yc, const arma::mat& y, double reg)
{
  int N = yc.n_rows;
  double ce_sum = -arma::accu (arma::log (yc.elem (arma::find (y == 1))));

  double data_loss = ce_sum / N;
  double reg_loss = 0.5 * reg * norms(nn);
  double loss = data_loss + reg_loss;
  // std::cout << "Loss: " << loss << "\n";
  return loss;
}

/*
 * Returns a vector of labels for each row vector in the input
 */
void predict (TwoLayerNet &nn, const arma::mat& X, arma::mat& label)
{
  struct cache fcache;
  feedforward (nn, X, fcache);
  label.set_size (X.n_rows);

  for (int i = 0; i < X.n_rows; ++i) {
    arma::uword row, col;
    fcache.yc.row(i).max (row, col);
    label(i) = col;
  }
}

/* 
 * Computes the numerical gradient
 */
void numgrad (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double reg, struct grads& numgrads)
{
  double h = 0.00001;
  struct cache numcache;
  numgrads.dW.resize(nn.num_layers);
  numgrads.db.resize(nn.num_layers);

  for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.dW[i].resize (nn.W[i].n_rows, nn.W[i].n_cols);
    for (int j = 0; j < nn.W[i].n_rows; ++j) {
      for (int k = 0; k < nn.W[i].n_cols; ++k) {
        double oldval = nn.W[i](j,k);
        nn.W[i](j, k) = oldval + h;
        feedforward (nn, X, numcache);
        double fxph = loss (nn, numcache.yc, y, reg);
        nn.W[i](j, k) = oldval - h;
        feedforward (nn, X, numcache);
        double fxnh = loss (nn, numcache.yc, y, reg);
        numgrads.dW[i](j, k) = (fxph - fxnh) / (2*h);
        nn.W[i](j, k) = oldval;
      }
    }
  }

   for (int i = 0; i < nn.num_layers; ++i) {
    numgrads.db[i].resize (nn.b[i].n_rows, nn.b[i].n_cols);
    for (int j = 0; j < nn.b[i].size(); ++j) {
      double oldval = nn.b[i](j);
      nn.b[i](j) = oldval + h;
      feedforward (nn, X, numcache);
      double fxph = loss (nn, numcache.yc, y, reg);
      nn.b[i](j) = oldval - h;
      feedforward (nn, X, numcache);
      double fxnh = loss (nn, numcache.yc, y, reg);
      numgrads.db[i](j) = (fxph - fxnh) / (2*h);
      nn.b[i](j) = oldval;
    }
  }
}

/*
 * Train the neural network &nn
 */
void train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every, int debug)
{
  int N = X.n_rows;
  int iter = 0;
  int print_flag = 0;

  for (int epoch = 0 ; epoch < epochs; ++epoch) {
    int num_batches = (int) ceil ( N / (float) batch_size);    

    for (int batch = 0; batch < num_batches; ++batch) {
      int last_row = std::min((batch + 1)*batch_size-1, N-1);
      arma::mat X_batch = X.rows (batch * batch_size, last_row);
      arma::mat y_batch = y.rows (batch * batch_size, last_row);

      struct cache bpcache;
      feedforward (nn, X_batch, bpcache);
      
      struct grads bpgrads;
      backprop (nn, y_batch, reg, bpcache, bpgrads);

      if (print_every > 0 && iter % print_every == 0) {
       if (grad_check) {
          struct grads numgrads;
          numgrad (nn, X_batch, y_batch, reg, numgrads);
          assert (gradcheck (numgrads, bpgrads));
        }
        std::cout << "Loss at iteration " << iter << " of epoch " << epoch << "/" << epochs << " = " << loss (nn, bpcache.yc, y_batch, reg) << "\n";
      }

      // Gradient descent step
      for (int i = 0; i < nn.W.size(); ++i) {
        nn.W[i] -= learning_rate * bpgrads.dW[i];
      }

      for (int i = 0; i < nn.b.size(); ++i) {
        nn.b[i] -= learning_rate * bpgrads.db[i];
      }

      /* Debug routine runs only when debug flag is set. If print_every is zero, it saves
         for the first batch of each epoch to avoid saving too many large files.
         Note that for the first time, you have to run debug and serial modes together.
         This will run the following function and write out files to CPUmats folder.
         In the later runs (with same parameters), you can use just the debug flag to 
         output diff b/w CPU and GPU without running CPU version */
      if(print_every <= 0)
        print_flag = batch == 0;
      else
        print_flag = iter % print_every == 0;

      if(debug && print_flag)
        write_cpudata_tofile(nn, iter);

      iter++;
    }
  }    
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation 
 * should mainly be in this function.
 */
void parallel_train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every, int debug)
{
  int rank, num_procs;
  MPI_SAFE_CALL (MPI_Comm_size (MPI_COMM_WORLD, &num_procs));
  MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));

  // TRANSPOSED
  arma::mat X_t = X.t();
  arma::mat y_t = y.t();
  int N = (rank == 0) ? X_t.n_cols : 0;

  // dimensions
  int n_images = batch_size / num_procs;
  int n_0 = nn.H[0];
  int n_1 = nn.H[1];
  int n_2 = nn.H[2];
  // new matrices 
  arma::mat Db0_t(n_1, n_images);
  arma::mat Db1_t(n_2, n_images);
  // new matrices to use
  arma::mat DW0(n_1, n_0);
  arma::mat DW1(n_2, n_1);
  arma::mat Db0(nn.b[0]);
  arma::mat Db1(nn.b[1]);
  // sizes
  int X_size = n_images * n_0;
  int y_size = n_images * n_2;
  int W0_size = n_1 * n_0;
  int W1_size = n_2 * n_1;
  int b0_size = n_1 * n_images;
  int b1_size = n_2 * n_images;
  // mallocs
  double* X_batch_buffer = (double*) malloc(sizeof(double) * X_size);
  double* y_batch_buffer = (double*) malloc(sizeof(double) * y_size);
  double* X_batch_mem;// = (double*) malloc(sizeof(double) * X_size);
  double* y_batch_mem;// = (double*) malloc(sizeof(double) * y_size);
  double* DW0_local = (double*) malloc(sizeof(double) * W0_size);
  double* DW1_local = (double*) malloc(sizeof(double) * W1_size);
  double* Db0_t_local = (double*) malloc(sizeof(double) * b0_size);
  double* Db1_t_local = (double*) malloc(sizeof(double) * b1_size);
  // arma pointers
  double* W0_mem = nn.W[0].memptr();
  double* W1_mem = nn.W[1].memptr();
  double* DW0_mem = DW0.memptr();
  double* DW1_mem = DW1.memptr();
  double* Db0_t_mem = Db0_t.memptr();
  double* Db1_t_mem = Db1_t.memptr();
  // broad cast
  MPI_SAFE_CALL (MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD));
// MPI_Bcast(
//     void* data,
//     int count,
//     MPI_Datatype datatype,
//     int root,
//     MPI_Comm communicator);

  std::ofstream error_file;
  error_file.open("Outputs/CpuGpuDiff.txt");
  int print_flag = 0;

  /* HINT: You can obtain a raw pointer to the memory used by Armadillo Matrices
     for storing elements in a column major way. Or you can allocate your own array
     memory space and store the elements in a row major way. Remember to update the
     Armadillo matrices in TwoLayerNet &nn of rank 0 before returning from the function. */
  
  /* iter is a variable used to manage debugging. It increments in the inner loop
     and therefore goes from 0 to epochs*num_batches */
  int iter = 0;

  for (int epoch = 0; epoch < epochs; ++epoch) {
    int num_batches = (N + batch_size - 1) / batch_size;
    for (int batch = 0; batch < num_batches; ++batch) {

      // if (rank == 0) {
        // subset by row number
        int last_col = std::min((batch + 1)*batch_size-1, N-1);
        arma::mat X_batch = X_t.cols (batch * batch_size, last_col);
        arma::mat y_batch = y_t.cols (batch * batch_size, last_col);
        X_batch_mem = X_batch.memptr();
        y_batch_mem = y_batch.memptr();
      // }

      /*
       * Possible Implementation:
       * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
       * 2. compute each sub-batch of images' contribution to network coefficient updates
       * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
       * 4. update local network coefficient at each node
       */

      // update every loop
      arma::mat b0_t = arma::repmat(nn.b[0].t(), 1, n_images);
      arma::mat b1_t = arma::repmat(nn.b[1].t(), 1, n_images);
      double* b0_t_mem = b0_t.memptr();
      double* b1_t_mem = b1_t.memptr();

      // scatter to all GPUS
      MPI_SAFE_CALL(MPI_Scatter(X_batch_mem, X_size, MPI_DOUBLE, X_batch_buffer, 
                  X_size, MPI_DOUBLE, 0, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Scatter(y_batch_mem, y_size, MPI_DOUBLE, y_batch_buffer, 
                  y_size, MPI_DOUBLE, 0, MPI_COMM_WORLD));

      // this function will call kernels to feedforward and backprop on the scattered chunk of data on GPU
      // it also applies the gradient descent 
      int gpu_success = gpu_train(X_batch_buffer, y_batch_buffer, W0_mem, W1_mem, b0_t_mem, b1_t_mem, 
                                  DW0_local, DW1_local, Db0_t_local, Db1_t_local,
                                  n_images, n_0, n_1, n_2, reg, learning_rate);

      // MPI_Allreduce() on DW0, DW1, Db0_t, Db1_t
      MPI_SAFE_CALL(MPI_Allreduce(DW0_local, DW0_mem, W0_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(DW1_local, DW1_mem, W1_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(Db0_t_local, Db0_t_mem, b0_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
      MPI_SAFE_CALL(MPI_Allreduce(Db1_t_local, Db1_t_mem, b1_size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));

      // b0 and b1 need to be compressed back to a single vector
      Db0 = arma::sum(Db0_t, 1).t() * (1.0/(double)n_images);
      Db1 = arma::sum(Db1_t, 1).t() * (1.0/(double)n_images);

      // UPDATES
      nn.W[0] -= learning_rate * DW0;
      nn.W[1] -= learning_rate * DW1; 
      nn.b[0] -= learning_rate * Db0;
      nn.b[1] -= learning_rate * Db1;

      if(print_every <= 0)
        print_flag = batch == 0;
      else
        print_flag = iter % print_every == 0;

      /* Following debug routine assumes that you have alread updated the arma 
         matrices in the TwoLayerNet nn.  */
      if(debug && rank == 0 && print_flag)
        write_diff_gpu_cpu(nn, iter, error_file);

      iter++;
    }
  }

  // freeze
  free(X_batch_buffer);
  free(y_batch_buffer);
  // free(X_batch_mem);
  // free(y_batch_mem);
  free(DW0_local);
  free(DW1_local);
  free(Db0_t_local);
  free(Db1_t_local);

  error_file.close();
}