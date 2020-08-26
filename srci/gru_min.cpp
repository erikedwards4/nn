//Includes
#include "gru_min.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 3u, O = 1u;
size_t dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does GRU (gated recurrent unit) model for driving inputs in X.\n";
descr += "\n";
descr += "X has size 2NxT or Tx2N, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "X has 2N time-series because it stacks the 2 driving inputs into one matrix:\n";
descr += "For dim==0, X = [X; Xf], and for dim==1, X = [X Xf]. \n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length 2N [default=0].\n";
descr += "\n";
descr += "For dim=0, F[:,t] = sig{X[N:2N-1,t] + Y[:,t-1]*Uf} \n";
descr += "           H[:,t] = F[:,t].*Y[:,t-1] \n";
descr += "           Y[:,t] = H[:,t] + (1-F[:,t]).*tanh{X[0:N-1,t] + H[:,t]*U} \n";
descr += "with sizes X: 2N x T \n";
descr += "           U:  N x N \n";
descr += "           Uf: N x N \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, F[t,:] = sig{X[t,N:2N-1] + Y[t-1,:]*Uf} \n";
descr += "           H[t,:] = F[t,:].*Y[t-1,:] \n";
descr += "           Y[t,:] = H[t,:] + (1-F[t,:]).*tanh{X[t,N:2N-1] + H[t,:]*U} \n";
descr += "with sizes X:  T x 2N \n";
descr += "           U:  N x N \n";
descr += "           Uf: N x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ gru_min X U Uf -o Y \n";
descr += "$ gru_min X U Uf > Y \n";
descr += "$ cat X | gru_min - U Uf > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,U,Uf)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "inputs 1-3 (X,U,Uf) must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Uf) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) must be a matrix" << endl; return 1; }
if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Uf) must be a matrix" << endl; return 1; }
if (i2.R!=i3.R || i2.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2 and 3 (U, Uf) must have the same size" << endl; return 1; }
if (i2.R!=i2.C || i3.R!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2 and 3 (U, Uf) must be square" << endl; return 1; }
if (dim==0u && i1.R%2u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be even for dim=0" << endl; return 1; }
if (dim==1u && i1.C%2u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be even for dim=1" << endl; return 1; }
if (dim==0u && i1.R!=2u*i2.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2 and 3 (U, Uf) must have size NxN" << endl; return 1; }
if (dim==1u && i1.C!=2u*i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2 and 3 (U, Uf) must have size NxN" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0u) ? i1.R/2u : i1.R;
o1.C = (dim==1u) ? i1.C/2u : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;

//Process
if (i1.T==1u)
{
    float *X, *U, *Uf; //*Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { U = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (U)" << endl; return 1; }
    try { Uf = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Uf)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(U),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (U)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(Uf),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Uf)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    //if (codee::gru_min_s(Y,X,U,Uf,N,T,i1.iscolmajor(),dim))
    if (codee::gru_min_inplace_s(X,U,Uf,N,T,i1.iscolmajor(),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        //try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        if ((dim==0u && o1.isrowmajor()) || (dim==1u && o1.iscolmajor()))
        {
            try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        else
        {
            float *Y;
            try { Y = new float[o1.N()]; }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
            for (size_t t=0u; t<T; ++t)
            {
                try { cblas_scopy((int)N,&X[2u*t*N],1,&Y[t*N],1); }
                catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
            }
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
            delete[] Y;
        }
    }
    delete[] X; delete[] U; delete[] Uf; //delete[] Y;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish
