//Includes
//#include <chrono>
#include "gru.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 4, O = 1;
int dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does GRU (gated recurrent unit) model for driving inputs X, Xr, Xz.\n";
descr += "This is the fully gated unit with 3 inputs (see also gru_min2).\n";
descr += "\n";
descr += "Input X has size 3NxT or Tx3N, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "X has 3N time-series because it stacks 3 driving inputs into one matrix:\n";
descr += "For dim==0, X = [X; Xr; Xz], and for dim==1, X = [X Xr Xz]. \n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length 3N [default=0].\n";
descr += "\n";
descr += "For dim=0, R[:,t] = sig{Xr[:,t] + Ur*Y[:,t-1]} \n";
descr += "           Z[:,t] = sig{Xz[:,t] + Uz*Y[:,t-1]} \n";
descr += "           H[:,t] = R[:,t].*Y[:,t-1] \n";
descr += "           Y[:,t] = Z[:,t].*Y[:,t-1] + (1-Z[:,t]).*tanh{X[:,t] + U*H[:,t]} \n";
descr += "with sizes X, Xr, Xz: N x T \n";
descr += "           U, Ur, Uz: N x N \n";
descr += "           Y        : N x T \n";
descr += "\n";
descr += "For dim=1, R[t,:] = sig{Xr[t,:] + Y[t-1,:]*Ur} \n";
descr += "           Z[t,:] = sig{Xz[t,:] + Y[t-1,:]*Uz} \n";
descr += "           H[t,:] = R[t,:].*Y[t-1,:] \n";
descr += "           Y[t,:] = Z[t,:].*Y[t-1,:] + (1-Z[t,:]).*tanh{X[t,:] + H[t,:]*U} \n";
descr += "with sizes X, Xr, Xz: T x N \n";
descr += "           U, Ur, Uz: N x N \n";
descr += "           Y        : T x N \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ gru X U Ur Uz -o Y \n";
descr += "$ gru X U Ur Uz > Y \n";
descr += "$ cat X | gru - U Ur Uz > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,U,Ur,Uz)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T)
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "inputs 1-4 (X,U,Ur,Uz) must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Ur) found to be empty" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Uz) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) must be a matrix" << endl; return 1; }
if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Ur) must be a matrix" << endl; return 1; }
if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Uz) must be a matrix" << endl; return 1; }
if (i2.R!=i3.R || i2.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (U, Ur, Uz) must have the same size" << endl; return 1; }
if (i2.R!=i4.R || i2.C!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (U, Ur, Uz) must have the same size" << endl; return 1; }
if (i2.R!=i2.C || i3.R!=i3.C || i4.R!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (U, Ur, Uz) must be square" << endl; return 1; }
if (dim==0 && i1.R%3u) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be multiple of 3 for dim=0" << endl; return 1; }
if (dim==1 && i1.C%3u) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be multiple of 3 for dim=1" << endl; return 1; }
if (dim==0 && i1.R!=3u*i2.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (U, Ur, Uz) must have size NxN" << endl; return 1; }
if (dim==1 && i1.C!=3u*i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 2-4 (U, Ur, Uz) must have size NxN" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? i1.R/3u : i1.R;
o1.C = (dim==1) ? i1.C/3u : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);

//Process
if (i1.T==1)
{
    float *X, *U, *Ur, *Uz, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { U = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (U)" << endl; return 1; }
    try { Ur = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Ur)" << endl; return 1; }
    try { Uz = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (Uz)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(U),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (U)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(Ur),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Ur)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(Uz),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (Uz)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    if (openn::gru_s(Y,X,U,Ur,Uz,N,T,dim,i1.iscolmajor()))
    //if (openn::gru_inplace_s(X,U,Ur,Uz,N,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        // if ((dim==0 && o1.isrowmajor()) || (dim==1 && o1.iscolmajor()))
        // {
        //     try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        //     catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        // }
        // else
        // {
        //     float *Y;
        //     try { Y = new float[o1.N()]; }
        //     catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        //     for (int t=0; t<T; t++)
        //     {
        //         try { cblas_scopy(N,&X[3*t*N],1,&Y[t*N],1); }
        //         catch(...) { cerr << progstr+": " << __LINE__ << errstr << "problem copying to output file (Y)" << endl; return 1; }
        //     }
        //     try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        //     catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        //     delete[] Y;
        // }
    }
    delete[] X; delete[] U; delete[] Ur; delete[] Uz; delete[] Y;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish

