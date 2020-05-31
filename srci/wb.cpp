//Includes
#include "wb.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 3, O = 1;
int dim;

//Description
string descr;
descr += "Neuron Input method.\n";
descr += "Applies weights (w) plus bias (b) along rows or cols of X.\n";
descr += "The weights are entered as a vector W with length matched to X.\n";
descr += "The bias is entered as a single number b.\n";
descr += "\n";
descr += "The output Y is a vector with: \n";
descr += "Y = W'*X + b,  for dim=0.\n";
descr += "Y = X*W + b ,  for dim=1.\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) [default=0].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ wb X W b -o Y \n";
descr += "$ wb X W b > Y \n";
descr += "$ cat X | wb - W b > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,W,b)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (b) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) must be a vector" << endl; return 1; }
if (!i3.isscalar()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (b) must be a scalar" << endl; return 1; }
if (dim==0 && i1.R!=i2.N()) { cerr << progstr+": " << __LINE__ << errstr << "length W must equal nrows X for dim=0" << endl; return 1; }
if (dim==1 && i1.C!=i2.N()) { cerr << progstr+": " << __LINE__ << errstr << "length W must equal ncols X for dim=1" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? 1 : i1.R;
o1.C = (dim==1) ? 1 : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1)
{
    float *X, *W, *b, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { b = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (b)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(b),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (b)" << endl; return 1; }
    if (openn::wb_s(Y,X,i1.iscolmajor(),int(i1.R),int(i1.C),W,b,dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}
else if (i1.T==101)
{
    float *X, *W, *b, *Y;
    try { X = new float[2*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[2*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { b = new float[2*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (b)" << endl; return 1; }
    try { Y = new float[2*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(b),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (b)" << endl; return 1; }
    if (openn::wb_c(Y,X,i1.iscolmajor(),int(i1.R),int(i1.C),W,b,dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish

