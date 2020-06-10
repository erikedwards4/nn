//Includes
#include "maxout.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 1, O = 1;
int dim, M, N, T;

//Description
string descr;
descr += "Layer activation function.\n";
descr += "Gets maxout function for each neuron of X.\n";
descr += "This is not just the max along rows or cols of X.\n";
descr += "Rather, it is the max separately for each of N neurons.\n";
descr += "\n";
descr += "Use -m (--M) to give the number of inputs per neuron within X.\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of the neurons [default=0].\n";
descr += "\n";
descr += "For dim=0: X has size MN x T, and Y has size N x T. \n";
descr += "For dim=1: X has size T x MN, and Y has size T x N. \n";
descr += "\n";
descr += "To achieve the full maxout unit, X should join M separate \n";
descr += "applications of the linear input stage (weights and biases).\n";
descr += "\n";
descr += "For complex input X, output Y is complex with elements having max absolute values.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ maxout -m2 X -o Y \n";
descr += "$ maxout -m3 X > Y \n";
descr += "$ cat X | maxout -m2 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_int    *a_m = arg_intn("m","M","<uint>",0,1,"number of inputs per neuron [default=1]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Get M
if (a_m->count==0) { M = 1; }
else if (a_m->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "M must be positive" << endl; return 1; }
else { M = a_m->ival[0]; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
if (dim==0 && i1.R%uint(M)) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be a multiple of M for dim=0" << endl; return 1; }
if (dim==1 && i1.C%uint(M)) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be a multiple of M for dim=1" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? i1.R/uint(M) : i1.R;
o1.C = (dim==1) ? i1.C/uint(M) : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);

//Process
if (i1.T==1)
{
    float *X, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (openn::maxout_s(Y,X,N,T,M,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}
else if (i1.T==101)
{
    float *X, *Y;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (openn::maxout_c(Y,X,N,T,M,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}

//Finish

