//Includes
#include "linear.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 3, O = 1;
int dim, Ni, No, T;

//Description
string descr;
descr += "Linear Input method.\n";
descr += "Applies weights (W) plus bias (B) to matrix X by linear method.\n";
descr += "The weights are entered as a matrix W.\n";
descr += "The biases are entered as a vector B with length No.\n";
descr += "\n";
descr += "For dim=0: Y = W*X + B \n";
descr += "with sizes X:  Ni x T  \n";
descr += "           W:  No x Ni \n";
descr += "           B:  No x 1  \n";
descr += "           Y:  No x T  \n";
descr += "\n";
descr += "For dim=1: Y = X*W + B \n";
descr += "with sizes X:  T x Ni  \n";
descr += "           W: Ni x No  \n";
descr += "           B:  1 x No  \n";
descr += "           Y:  T x No  \n";
descr += "\n";
descr += "where Ni is num inputs, No is num outputs, and T is num time-points.\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) [default=0].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ linear X W B -o Y \n";
descr += "$ linear X W B > Y \n";
descr += "$ cat X | linear - W B > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,W,B)");
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
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) must be a matrix" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) must be a vector" << endl; return 1; }
if (dim==0 && i1.R!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "ncols W must equal nrows X for dim=0" << endl; return 1; }
if (dim==1 && i1.C!=i2.R) { cerr << progstr+": " << __LINE__ << errstr << "nrows W must equal ncols X for dim=1" << endl; return 1; }
if (dim==0 && i3.N()!=i2.R) { cerr << progstr+": " << __LINE__ << errstr << "length B must equal nrows W for dim=0" << endl; return 1; }
if (dim==1 && i3.N()!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "length B must equal ncols W for dim=1" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? i2.R : i1.R;
o1.C = (dim==1) ? i2.C : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep
Ni = (dim==0) ? int(i1.R) : int(i1.C);
No = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(i1.C) : int(i1.R);

//Process
if (i1.T==1)
{
    float *X, *W, *B, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { B = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(B),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (openn::linear_s(Y,X,W,B,Ni,No,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] W; delete[] B; delete[] Y;
}
else if (i1.T==101)
{
    float *X, *W, *B, *Y;
    try { X = new float[2*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[2*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { B = new float[2*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[2*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(B),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (openn::linear_c(Y,X,W,B,Ni,No,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] W; delete[] B; delete[] Y;
}

//Finish

