//Includes
#include "grossberg2.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 8u, O = 1u;
size_t dim, N, T;
double fs;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does middle stage of Grossberg model for each row or col of Xe and Xi.\n";
descr += "which are separate excitatory and inhibitory driving inputs.\n";
descr += "\n";
descr += "The soma model is temporal integration (1st-order IIR filter, time-constant tau),\n";
descr += "along with a subtractive feedback with gain alpha,\n";
descr += "along with adaptive gain control with parameters beta and gamma.\n";
descr += "\n";
descr += "Xe and Xi have size NxT or TxN, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "For dim=0, Y[n,t] = a[n]*Y[n,t-1] + b[n]*(g[n,t]*X[n,t]-alpha[n]*Y[n,t-1]). \n";
descr += "with sizes X:  N x T \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, Y[t,n] = a[n]*Y[t-1,n] + b[n]*(g[t,n]*X[t,n]-alpha[n]*Y[t-1,n]). \n";
descr += "with sizes X:  T x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "where a[n] = exp(-1/(fs*taus[n])) and b[n] = 1 - a[n].\n";
descr += "and g[n,t] = gamma[n] - beta[n]*Y[n,t-1].\n";
descr += "\n";
descr += "Use -s (--fs) to give the sample rate of X in Hz [default=10000].\n";
descr += "\n";
descr += "Enter a vector of N taus as the 3rd input.\n";
descr += "Or enter a single tau as the 3rd input, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter a vector of N alphas as the 4th input.\n";
descr += "Or enter a single alpha as the 4th input, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter vectors of N betas as the 5th and 6th inputs (e and i separately).\n";
descr += "Or enter a betae and betai as the 5th and 6th inputs, to be used by all N neurons.\n";
descr += "\n";
descr += "Enter vector of N gammas as the 7th and 8th inputs (e and i separately).\n";
descr += "Or enter a gammae and gammai as the 7th and 8th inputs, to be used by all N neurons.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ grossberg2 Xe Xi tau alpha betae betai gammae gammai -o Y \n";
descr += "$ grossberg2 Xe Xi tau alpha betae betai gammae gammai > Y \n";
descr += "$ cat X | grossberg2 - tau alpha betae betai gammae gammai > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (Xe,Xi,tau,alpha,betae,betai,gammae,gammai)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_dbl   *a_fs = arg_dbln("s","fs","<dbl>",0,1,"sample rate of Xe and Xi [default=10000]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = i1.isvec() ? i1.nonsingleton1() : 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>1u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Get fs
fs = (a_fs->count>0) ? a_fs->dval[0] : 10000.0;
if (fs<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "fs (sample rate) must be positive" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have same data type" << endl; return 1; }
if (i1.T!=i5.T || i1.T!=i6.T || i1.T!=i7.T || i1.T!=i8.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have same data type" << endl; return 1; }
if (!i1.isvec() && i1.iscolmajor()!=i2.iscolmajor())
{ cerr << progstr+": " << __LINE__ << errstr << "inputs 1 and 2 (Xe,Xi) must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) must be a matrix" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) found to be empty" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) must be a matrix" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (tau) found to be empty" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (tau) must be a vector or scalar" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (alpha) found to be empty" << endl; return 1; }
if (!i4.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (alpha) must be a vector or scalar" << endl; return 1; }
if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (betae) found to be empty" << endl; return 1; }
if (!i5.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (betae) must be a vector or scalar" << endl; return 1; }
if (i6.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (betai) found to be empty" << endl; return 1; }
if (!i6.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (betai) must be a vector or scalar" << endl; return 1; }
if (i7.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (gammae) found to be empty" << endl; return 1; }
if (!i7.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (gammae) must be a vector or scalar" << endl; return 1; }
if (i8.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (gammai) found to be empty" << endl; return 1; }
if (!i8.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (gammai) must be a vector or scalar" << endl; return 1; }
if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1 (Xe) and 2 (Xi) must have same size" << endl; return 1; }
if (i3.N()!=1u)
{
    if ((dim==0u && i3.N()!=i1.R) || (dim==1u && i3.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length tau must equal N (num neurons)" << endl; return 1; }
}
if (i4.N()!=1u)
{
    if ((dim==0u && i4.N()!=i1.R) || (dim==1u && i4.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length alpha must equal N (num neurons)" << endl; return 1; }
}
if (i5.N()!=1u)
{
    if ((dim==0u && i5.N()!=i1.R) || (dim==1u && i5.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length betae must equal N (num neurons)" << endl; return 1; }
}
if (i6.N()!=1u)
{
    if ((dim==0u && i6.N()!=i1.R) || (dim==1u && i6.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length betai must equal N (num neurons)" << endl; return 1; }
}
if (i7.N()!=1u)
{
    if ((dim==0u && i7.N()!=i1.R) || (dim==1u && i7.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length gammae must equal N (num neurons)" << endl; return 1; }
}
if (i8.N()!=1u)
{
    if ((dim==0u && i8.N()!=i1.R) || (dim==1u && i8.N()!=i1.C))
    { cerr << progstr+": " << __LINE__ << errstr << "length gammai must equal N (num neurons)" << endl; return 1; }
}

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0u) ? o1.R : o1.C;
T = (dim==0u) ? o1.C : o1.R;
if (T<2u) { cerr << progstr+": " << __LINE__ << errstr << "num time points must be > 1" << endl; return 1; }

//Process
if (i1.T==1u)
{
    float *Xe, *Xi, *tau, *alpha, *betae, *betai, *gammae, *gammai;
    try { Xe = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (Xe)" << endl; return 1; }
    try { Xi = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xi)" << endl; return 1; }
    try { tau = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (tau)" << endl; return 1; }
    try { alpha = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (alpha)" << endl; return 1; }
    try { betae = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (betae)" << endl; return 1; }
    try { betai = new float[i6.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (betai)" << endl; return 1; }
    try { gammae = new float[i7.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 7 (gammae)" << endl; return 1; }
    try { gammai = new float[i8.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 8 (gammai)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(Xe),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (Xe)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(Xi),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xi)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(tau),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (tau)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(alpha),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (alpha)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(betae),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (betae)" << endl; return 1; }
    try { ifs6.read(reinterpret_cast<char*>(betai),i6.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (betai)" << endl; return 1; }
    try { ifs7.read(reinterpret_cast<char*>(gammae),i7.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 7 (gammae)" << endl; return 1; }
    try { ifs8.read(reinterpret_cast<char*>(gammai),i8.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 8 (gammai)" << endl; return 1; }
    if (i3.N()==1u) { for (size_t n=1u; n<N; ++n) { tau[n] = tau[0]; } }
    if (i4.N()==1u) { for (size_t n=1u; n<N; ++n) { alpha[n] = alpha[0]; } }
    if (i5.N()==1u) { for (size_t n=1u; n<N; ++n) { betae[n] = betae[0]; } }
    if (i6.N()==1u) { for (size_t n=1u; n<N; ++n) { betai[n] = betai[0]; } }
    if (i7.N()==1u) { for (size_t n=1u; n<N; ++n) { gammae[n] = gammae[0]; } }
    if (i8.N()==1u) { for (size_t n=1u; n<N; ++n) { gammai[n] = gammai[0]; } }
    if (codee::grossberg2_inplace_s(Xe,Xi,tau,alpha,betae,betai,gammae,gammai,N,T,i1.iscolmajor(),dim,float(fs)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Xe),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] Xe; delete[] Xi; delete[] tau; delete[] alpha; delete[] betae; delete[] betai; delete[] gammae; delete[] gammai;
}

//Finish
