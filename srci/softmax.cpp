//Includes
#include "softmax.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 1, O = 1;
int dim;

//Description
string descr;
descr += "Layer activation function.\n";
descr += "Gets softmax function for each row or col of X.\n";
descr += "The output Y has the same size as X, with:\n";
descr += "Y[r,c] = exp(X[r,c]) / sum(exp(X[:,c])),  for dim=0.\n";
descr += "Y[r,c] = exp(X[r,c]) / sum(exp(X[r,:])),  for dim=1.\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) [default=0].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ softmax X -o Y \n";
descr += "$ softmax X > Y \n";
descr += "$ cat X | softmax > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1)
{
    float *X;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (openn::softmax_inplace_s(X,i1.iscolmajor(),int(i1.R),int(i1.C),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

