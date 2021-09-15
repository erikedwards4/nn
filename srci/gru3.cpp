//Includes
//#include <chrono>
#include "gru3.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 6u, O = 1u;
size_t dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does GRU (gated recurrent unit) model for driving inputs X, Xr, Xz.\n";
descr += "This is the fully gated unit with 3 inputs (see also gru_min2).\n";
descr += "\n";
descr += "X is the usual input and Xr, Xz are the inputs for the reset and update gates.\n";
descr += "All have size NxT or TxN, where N is the number of neurons,\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
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
descr += "$ gru3 X Xr Xz U Ur Uz -o Y \n";
descr += "$ gru3 X Xr Xz U Ur Uz > Y \n";
descr += "$ cat X | gru3 - Xr Xz U Ur Uz > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,Xr,Xz,U,Ur,Uz)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T || i1.T!=i5.T || i1.T!=i6.T)
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor() || i1.iscolmajor()!=i5.iscolmajor() || i1.iscolmajor()!=i6.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xr) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xz) found to be empty" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (U) found to be empty" << endl; return 1; }
if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Ur) found to be empty" << endl; return 1; }
if (i6.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Uz) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xr) must be a matrix" << endl; return 1; }
if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xz) must be a matrix" << endl; return 1; }
if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (U) must be a matrix" << endl; return 1; }
if (!i5.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Ur) must be a matrix" << endl; return 1; }
if (!i6.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Uz) must be a matrix" << endl; return 1; }
if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-3 (X, Xr, Xz) must have the same size" << endl; return 1; }
if (i1.R!=i3.R || i1.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-3 (X, Xr, Xz) must have the same size" << endl; return 1; }
if (i4.R!=i5.R || i4.C!=i5.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have the same size" << endl; return 1; }
if (i4.R!=i6.R || i4.C!=i6.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have the same size" << endl; return 1; }
if (i4.R!=i4.C || i5.R!=i5.C || i6.R!=i6.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must be square" << endl; return 1; }
if (dim==0u && i1.R!=i4.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have size NxN" << endl; return 1; }
if (dim==1u && i1.C!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 4-6 (U, Ur, Uz) must have size NxN" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;

//Process
if (i1.T==1u)
{
    float *X, *Xr, *Xz, *U, *Ur, *Uz, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Xr = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xr)" << endl; return 1; }
    try { Xz = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Xz)" << endl; return 1; }
    try { U = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (U)" << endl; return 1; }
    try { Ur = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (Ur)" << endl; return 1; }
    try { Uz = new float[i6.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (Uz)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(Xr),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xr)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(Xz),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Xz)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(U),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (U)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(Ur),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (Ur)" << endl; return 1; }
    try { ifs6.read(reinterpret_cast<char*>(Uz),i6.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (Uz)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    if (codee::gru3_s(Y,X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
    //if (codee::gru3_inplace_s(X,Xr,Xz,U,Ur,Uz,N,T,i1.iscolmajor(),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Xr; delete[] Xz; delete[] U; delete[] Ur; delete[] Uz; delete[] Y;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish
