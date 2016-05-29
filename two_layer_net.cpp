#include "two_layer_net.h"

#include <armadillo>
#include "utils/common.h"
#include "gpu_func.h"
#include "mpi.h"
#include "iomanip"

#define BUGIDX 0
#define BUGINC 10

#define BUG_EPOCH 1
#define BUG_BATCH 1

// for debugging
arma::mat g_DW1;
arma::mat g_diff_t;
arma::mat g_Da1_t;
arma::mat g_Dz1_t;
arma::mat g_DW0;
arma::mat g_Db0;
arma::mat g_Db1;

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

  g_diff_t = diff.t();
  g_Db0 = bpgrads.db[0];
  g_Db1 = bpgrads.db[1];
  g_DW1 = bpgrads.dW[1];
  g_Dz1_t = dz1.t();
  g_DW0 = bpgrads.dW[0];

  // std::cout << "DW0\n" << g_DW0.col(BUGIDX) << std::endl;

  // std::cout << "DW1\n" << g_DW1.col(BUGIDX) << std::endl;

  // std::cout << "Db0\n" << g_Db0 << std::endl;

  // std::cout << "Db1\n" << g_Db1 << std::endl;

  // std::cout << "CPU z1_t" << std::endl;
  // arma::mat z1_t = bpcache.z[0].t();
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << z1_t.memptr()[i] << std::endl;
  // }

  // std::cout << "\nCPU a1_t" << std::endl;
  // arma::mat a1_t = bpcache.a[0].t();
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << a1_t.memptr()[i] << std::endl;
  // }

  // std::cout << "\nCPU diff.t" << std::endl;
  // arma::mat diff_t = diff.t();
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << diff_t.memptr()[i] << std::endl;
  // }

  // std::cout << "\nCPU W1" << std::endl;
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << nn.W[1].memptr()[i] << std::endl;
  // }

  // std::cout << "\nCPU dW[1] " <<  std::endl; 
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << bpgrads.dW[1].memptr()[i] << std::endl;
  // }

  // g_Da1_t = da1.t();
  // arma::mat da1_t = da1.t();
  // std::cout << "\nCPU Da1_t " <<  std::endl; 
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << da1_t.memptr()[i] << std::endl;
  // }

  // arma::mat dz1_t = dz1.t();
  // std::cout << "\nCPU Dz1_t " <<  std::endl; 
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << dz1_t.memptr()[i] << std::endl;
  // }

  // std::cout << "\nCPU W0" << std::endl;
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << nn.W[0].memptr()[i] << std::endl;
  // }

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

  for (int epoch = 0 ; epoch < BUG_EPOCH; ++epoch) {
  // for (int epoch = 0 ; epoch < epochs; ++epoch) {
    int num_batches = (int) ceil ( N / (float) batch_size);    

    for (int batch = 0; batch < BUG_BATCH; ++batch) {
    // for (int batch = 0; batch < num_batches; ++batch) {
      int last_row = std::min((batch + 1)*batch_size-1, N-1);
      arma::mat X_batch = X.rows (batch * batch_size, last_row);
      arma::mat y_batch = y.rows (batch * batch_size, last_row);

      arma::mat x_t = X_batch.t();
      arma::mat y_t = y_batch.t();

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

// X and y have been subdivided
int gpu_train(double* X, double* y, double* W0, double* W1, double* b0, double* b1, 
            double* DW0, double* DW1, double* Db0, double* Db1,
            const int n_images, const int n_0, const int n_1, const int n_2, 
            double reg, double learning_rate, int batch_size) {

  // create pointers
  double* d_X;
  double* d_y;
  double* d_W0;
  double* d_W1;
  double* d_b0;
  double* d_b1;
  // data only on device
  double* d_a1;
  double* d_a2;
  double* d_z1;
  double* d_z2;
  double* d_DW0;
  double* d_DW1;
  double* d_Db0;
  double* d_Db1;
  double* d_Da1;
  double* d_Dz1;

  // calc sizes
  const unsigned int X_size = n_images * n_0; // 800 x 784
  const unsigned int y_size = n_images * n_2; // 800 x 10
  const unsigned int W0_size = n_1 * n_0; // 100 x 784
  const unsigned int W1_size = n_2 * n_1; // 10 x 100
  const unsigned int b0_size = n_images * n_1; // 800 x 100
  const unsigned int b1_size = n_images * n_2; // 800 x 10
  const unsigned int a1_size = n_images * n_1; // 800 x 100
  const unsigned int a2_size = n_images * n_2; // 800 x 10
  const unsigned int z1_size = n_images * n_1; // 800 x 100
  const unsigned int z2_size = n_images * n_2; // 800 x 10

  // malloc
  checkCudaErrors(cudaMalloc(&d_X, X_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_y, y_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_W0, W0_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_W1, W1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_b0, b0_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_b1, b1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_a1, a1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_a2, a2_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z1, z1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_z2, z2_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_DW0, W0_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_DW1, W1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Db0, b0_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Db1, b1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Da1, a1_size * sizeof(double)));
  checkCudaErrors(cudaMalloc(&d_Dz1, z1_size * sizeof(double)));

  // memcpy
  checkCudaErrors(cudaMemcpy(d_X, X, X_size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_y, y, y_size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_W0, W0, W0_size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_W1, W1, W1_size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b0, b0, b0_size * sizeof(double), cudaMemcpyHostToDevice));
  checkCudaErrors(cudaMemcpy(d_b1, b1, b1_size * sizeof(double), cudaMemcpyHostToDevice));

  // feedforward steps to calc a1, a2, z1, z2 all on device
  // z1.T = W0 * X.T + b0.T
  myGEMM_no(d_W0, d_X, d_b0, d_z1, 1, 1, n_1, n_images, n_0);

  // arma::mat z1_t = arma::mat(n_1, n_images);
  // cudaMemcpy(z1_t.memptr(), d_z1, sizeof(double)*z1_size, cudaMemcpyDeviceToHost);
  
  // a1.T = sigmoid(z1.T)
  sigmoid_GPU(d_z1, d_a1, n_1, n_images);

  // arma::mat a1_t = arma::mat(n_1, n_images);
  // cudaMemcpy(a1_t.memptr(), d_a1, sizeof(double)*a1_size, cudaMemcpyDeviceToHost);

  // z2.T = W1 * a1.T + b1.T
  myGEMM_no(d_W1, d_a1, d_b1, d_z2, 1, 1, n_2, n_images, n_1);

  // arma::mat z2_t = arma::mat(n_2, n_images);
  // cudaMemcpy(z2_t.memptr(), d_z2, sizeof(double)*z2_size, cudaMemcpyDeviceToHost);

  // a2.T = (softmax(z2.T) - y) / n_images
  // a2.T now holds the CROSS ENTROPY
  softmax_GPU(d_z2, d_a2, d_y, n_2, n_images, batch_size);

  // arma::mat a2_t = arma::mat(n_2, n_images);
  // cudaMemcpy(a2_t.memptr(), d_a2, sizeof(double)*a2_size, cudaMemcpyDeviceToHost);

  // backprop steps to calc dW0-1 and db0-1 all on device
  // DW1 = CE * a1.T + reg * W1 where CE = "diff"
  myGEMM_no_tB(d_a2, d_a1, d_W1, d_DW1, 1, reg, n_2, n_1, n_images);

  // arma::mat DW1_mat = arma::mat(n_2, n_1);
  // cudaMemcpy(DW1_mat.memptr(), d_DW1, sizeof(double)*W1_size, cudaMemcpyDeviceToHost);

  // arma::mat W1_mat(W1, n_2, n_1, false);

  // Db1.T = a2.T ... do nothing
  // Da1.T = W1 * a2.T 
  myGEMM_no_na_tA(d_W1, d_a2, d_Da1, 1, n_1, n_images, n_2);

  // arma::mat Da1_t_mat = arma::mat(n_1, n_images);
  // cudaMemcpy(Da1_t_mat.memptr(), d_Da1, sizeof(double)*a1_size, cudaMemcpyDeviceToHost);

  // Dz1.T = Da1.T .* a1.T .* (1 - a1.T)
  Dz1_schur_GPU(d_Da1, d_a1, d_Dz1, n_1, n_images);

  // arma::mat Dz1_t_mat = arma::mat(n_1, n_images);
  // cudaMemcpy(Dz1_t_mat.memptr(), d_Dz1, sizeof(double)*z1_size, cudaMemcpyDeviceToHost);

  // arma::mat diff_Dz1_t = g_Dz1_t - Dz1_t_mat;
  // std::cout << "\ndiff Dz1_t " <<  std::endl; 
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << diff_Dz1_t.memptr()[i] << std::endl;
  // }  

  // DW0.T = Dz1.T * X.T + reg * W0
  myGEMM_no_tB(d_Dz1, d_X, d_W0, d_DW0, 1, reg, n_1, n_0, n_images);
  // Db0.T = Dz1.T ... do nothing

  // arma::mat X_t_mat = arma::mat(n_0, n_images);
  // cudaMemcpy(X_t_mat.memptr(), d_X, sizeof(double)*X_size, cudaMemcpyDeviceToHost);
  // arma::mat W0_mat = arma::mat(n_1, n_0);
  // cudaMemcpy(W0_mat.memptr(), d_W0, sizeof(double)*W0_size, cudaMemcpyDeviceToHost);

  // arma::mat DW0_mat = arma::mat(n_1, n_0);
  // cudaMemcpy(DW0_mat.memptr(), d_DW0, sizeof(double)*W0_size, cudaMemcpyDeviceToHost);

  // arma::mat diff_DW0 = g_DW0 - DW0_mat;
  // std::cout << "\n GPU diff DW0" << std::endl;
  // for (int i = BUGIDX; i < BUGIDX + BUGINC; ++i) {
  //   std::cout << i << ": " << diff_DW0.memptr()[i] << std::endl;
  // }


  // arma::mat Db0_t_mat = arma::mat(n_1, n_images);
  // cudaMemcpy(Db0_t_mat.memptr(), d_Dz1, sizeof(double)*b0_size, cudaMemcpyDeviceToHost);
  // arma::mat Db0_vec = arma::sum(Db0_t_mat, 1);
  // arma::mat diff_Db0_vec = g_Db0 - Db0_vec.t();
  // std::cout << "\n GPU diff Db0" << std::endl;
  // std::cout << diff_Db0_vec << std::endl;


  // memcpy
  checkCudaErrors(cudaMemcpy(DW0, d_DW0, W0_size * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(DW1, d_DW1, W1_size * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(Db0, d_Dz1, b0_size * sizeof(double), cudaMemcpyDeviceToHost));
  checkCudaErrors(cudaMemcpy(Db1, d_a2, b1_size * sizeof(double), cudaMemcpyDeviceToHost));

  // free!
  cudaFree(d_X);
  cudaFree(d_y);
  cudaFree(d_W0);
  cudaFree(d_W1);
  cudaFree(d_b0);
  cudaFree(d_b1);
  cudaFree(d_a1);
  cudaFree(d_a2);
  cudaFree(d_z1);
  cudaFree(d_z2);
  cudaFree(d_DW0);
  cudaFree(d_DW1);
  cudaFree(d_Db0);
  cudaFree(d_Db1);
  cudaFree(d_Da1);
  cudaFree(d_Dz1);

  return 0;
}

/*
 * Train the neural network &nn of rank 0 in parallel. Your MPI implementation 
 * should mainly be in this function.
 */
void parallel_train (TwoLayerNet &nn, const arma::mat& X, const arma::mat& y, double learning_rate, double reg, 
    const int epochs, const int batch_size, bool grad_check, int print_every, int debug)
{
  int rank;
  MPI_SAFE_CALL (MPI_Comm_rank (MPI_COMM_WORLD, &rank));

  // TRANSPOSED
  arma::mat X_t = X.t();
  arma::mat y_t = y.t();
  int N = (rank == 0) ? X.n_rows : 0;
  // broad cast
  MPI_SAFE_CALL (MPI_Bcast (&N, 1, MPI_INT, 0, MPI_COMM_WORLD));

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

  // hard coded for pseudo parallel
  int num_procs = 4;

  // for (int epoch = 0; epoch < epochs; ++epoch) {
  for (int epoch = 0; epoch < BUG_EPOCH; ++epoch) {

    int num_batches = (N + batch_size - 1) / batch_size;
    for (int batch = 0; batch < BUG_BATCH; ++batch) {
    // for (int batch = 0; batch < num_batches; ++batch) {
      // dimensions
      int n_images = batch_size / num_procs;
      int n_0 = nn.H[0];
      int n_1 = nn.H[1];
      int n_2 = nn.H[2];

      arma::mat X_batch;
      arma::mat y_batch;
      // double* X_batch_mem;
      // double* y_batch_mem;

      if (batch == num_batches - 1) {
        n_images = (N % batch_size) / num_procs;
      }

      // subset by row number
      int last_col = std::min((batch + 1)*batch_size-1, N-1);
      X_batch = X_t.cols (batch * batch_size, last_col);
      y_batch = y_t.cols (batch * batch_size, last_col);
      // X_batch_mem = X_batch.memptr();
      // y_batch_mem = y_batch.memptr();    

      // new matrices 
      arma::mat Db0_t = arma::zeros(n_1, n_images);
      arma::mat Db1_t = arma::zeros(n_2, n_images);
      arma::mat DW0 = arma::zeros(n_1, n_0);
      arma::mat DW1 = arma::zeros(n_2, n_1);
      arma::mat Db0;
      arma::mat Db1;

      // sizes
      int X_size = n_images * n_0;
      int y_size = n_images * n_2;
      int W0_size = n_1 * n_0;
      int W1_size = n_2 * n_1;
      int b0_size = n_1 * n_images;
      int b1_size = n_2 * n_images;
      // arma pointers
      double* W0_mem = nn.W[0].memptr();
      double* W1_mem = nn.W[1].memptr();
      double* DW0_mem = DW0.memptr();
      double* DW1_mem = DW1.memptr();
      double* Db0_t_mem = Db0_t.memptr();
      double* Db1_t_mem = Db1_t.memptr();

      // update every loop
      arma::mat b0_t = arma::repmat(nn.b[0].t(), 1, n_images);
      arma::mat b1_t = arma::repmat(nn.b[1].t(), 1, n_images);
      double* b0_t_mem = b0_t.memptr();
      double* b1_t_mem = b1_t.memptr();

        /*
         * Possible Implementation:
         * 1. subdivide input batch of images and `MPI_scatter()' to each MPI node
         * 2. compute each sub-batch of images' contribution to network coefficient updates
         * 3. reduce the coefficient updates and broadcast to all nodes with `MPI_Allreduce()'
         * 4. update local network coefficient at each node
         */

      for (int prank = 0; prank < num_procs; ++prank) {

        // mallocs
        arma::mat X_batch_buffer = X_batch.cols(n_images*prank, n_images*(prank+1)-1);
        arma::mat y_batch_buffer = y_batch.cols(n_images*prank, n_images*(prank+1)-1);
        arma::mat DW0_local = arma::zeros(n_1, n_0);
        arma::mat DW1_local = arma::zeros(n_2, n_1);
        arma::mat Db0_t_local = arma::zeros(n_1, n_images);
        arma::mat Db1_t_local = arma::zeros(n_2, n_images);

        // this function will call kernels to feedforward and backprop on the scattered chunk of data on GPU
        int gpu_success = gpu_train(X_batch_buffer.memptr(), y_batch_buffer.memptr(), W0_mem, W1_mem, b0_t_mem, b1_t_mem, 
                                    DW0_local.memptr(), DW1_local.memptr(), Db0_t_local.memptr(), Db1_t_local.memptr(),
                                    n_images, n_0, n_1, n_2, reg, learning_rate, n_images);

        DW0 += DW0_local;
        DW1 += DW1_local;
        Db0_t += Db0_t_local;
        Db1_t += Db1_t_local;

      }

      // b0 and b1 need to be compressed back to a single vector
      Db0 = arma::sum(Db0_t, 1).t();
      Db1 = arma::sum(Db1_t, 1).t();

      std::cout << " g_Db1\n" << g_Db1 << std::endl;
      std::cout << " pseudo parallel Db1\n" << Db1 / num_procs << std::endl;

      arma::mat DW0_diff = g_DW0 - (DW0/num_procs);
      arma::mat DW1_diff = g_DW1 - (DW1/num_procs);
      std::cout  << " Db0 diff\n" << g_Db0 - (Db0/num_procs) << std::endl;
      std::cout  << " Db1 diff\n" << g_Db1 - (Db1/num_procs) << std::endl;
      std::cout  << " DW0 diff\n" << DW0_diff.row(0) << std::endl;
      std::cout  << " DW1 diff\n" << DW1_diff.row(0) << std::endl;

      // UPDATES
      nn.W[0] -= DW0 * learning_rate / num_procs;
      nn.W[1] -= DW1 * learning_rate / num_procs; 
      nn.b[0] -= Db0 * learning_rate / num_procs;
      nn.b[1] -= Db1 * learning_rate / num_procs;

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



  error_file.close();
}