//Includes
#include "elman.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 5, O = 1;
int dim, N, T;

//Description
string descr;
descr += "ML Neuron Output Side.\n";
descr += "Recurrent NN (RNN) type.\n";
descr += "Does output side of Elman RNN for driving input X.\n";
descr += "\n";
descr += "X has size NxT or TxN, where N is the number of neurons\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "There is no exact specification of the Elman architecture in the literature.\n";
descr += "Here is given one possibility, where there are N neurons and N outputs.\n";
descr += "\n";
descr += "For dim=0, update equations are: H[:,t] = g(X[:,t] + U*H[:,t-1]) \n";
descr += "                                 Y[:,t] = g(W*H[:,t] + B) \n";
descr += "with sizes X:  N x T \n";
descr += "           U:  N x N \n";
descr += "           H:  N x T \n";
descr += "           W:  N x N \n";
descr += "           B:  N x 1 \n";
descr += "           Y:  N x T \n";
descr += "\n";
descr += "For dim=1, update equations are: H[t,:] = g(X[t,:] + H[t-1,:]*U) \n";
descr += "                                 Y[t,:] = g(H[t,:]*W + B) \n";
descr += "with sizes X:  T x N \n";
descr += "           U:  N x N \n";
descr += "           H:  T x N \n";
descr += "           W:  N x N \n";
descr += "           B:  1 x N \n";
descr += "           Y:  T x N \n";
descr += "\n";
descr += "H is input as a vector of length N to initialize the recursion.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ elman X U H W B -o Y \n";
descr += "$ elman X U H W B > Y \n";
descr += "$ cat X | elman - U H W B > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,U,H,W,B)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T || i1.T!=i5.T) { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "inputs 1, 2, 4 (X,U,W) must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (H) found to be empty" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (W) found to be empty" << endl; return 1; }
if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (B) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) must be a matrix" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (H) must be a vector" << endl; return 1; }
if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (W) must be a matrix" << endl; return 1; }
if (!i5.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (B) must be a vector" << endl; return 1; }
if (!i2.issquare()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (U) must be square" << endl; return 1; }
if (!i4.issquare()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (W) must be square" << endl; return 1; }
if (dim==0)
{
    if (i2.R!=i1.R) { cerr << progstr+": " << __LINE__ << errstr << "nrows U must equal nrows X for dim=0" << endl; return 1; }
    if (i3.N()!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "length H must equal ncols U for dim=0" << endl; return 1; }
    if (i4.C!=i3.N()) { cerr << progstr+": " << __LINE__ << errstr << "ncols W must equal length H for dim=0" << endl; return 1; }
    if (i5.N()!=i4.R) { cerr << progstr+": " << __LINE__ << errstr << "length B must equal nrows W for dim=0" << endl; return 1; }
}
if (dim==1)
{
    if (i2.C!=i1.C) { cerr << progstr+": " << __LINE__ << errstr << "ncols U must equal ncols X for dim=1" << endl; return 1; }
    if (i3.N()!=i2.R) { cerr << progstr+": " << __LINE__ << errstr << "length H must equal nrows U for dim=1" << endl; return 1; }
    if (i4.R!=i3.N()) { cerr << progstr+": " << __LINE__ << errstr << "nrows W must equal length H for dim=1" << endl; return 1; }
    if (i5.N()!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "length B must equal ncols W for dim=1" << endl; return 1; }
}

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);

//Process
if (i1.T==1)
{
    float *X, *U, *H, *W, *B, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { U = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (U)" << endl; return 1; }
    try { H = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (H)" << endl; return 1; }
    try { W = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (W)" << endl; return 1; }
    try { B = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (B)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(U),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (U)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(H),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (H)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(W),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (W)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(B),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (B)" << endl; return 1; }
    if (openn::elman_s(Y,X,U,H,W,B,N,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] U; delete[] H; delete[] W; delete[] B; delete[] Y;
}

//Finish

