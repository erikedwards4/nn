//Includes
#include "join3.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 3, O = 1;
int dim;

//Description
string descr;
descr += "Joins 3 inputs X1, X2, X3 into 1 output Y.\n";
descr += "\n";
descr += "Use -d (--dim) to give the dimension (axis) [default=0].\n";
descr += "Use -d0 to work along cols --> Y has size R1+R2+R3 x C.\n";
descr += "Use -d1 to work along rows --> Y has size R x C1+C2+C3.\n";
descr += "\n";
descr += "For dim=0, num cols X1, X2, X3 must be equal (C1=C2=C3).\n";
descr += "For dim=1, num rows X1, X2, X3 must be equal (R1=R2=R3).\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ join3 X1 X2 X3 -o Y \n";
descr += "$ join3 X1 X2 X3 > Y \n";
descr += "$ join3 -d1 X1 X2 X3 -o Y \n";
descr += "$ cat X1 | join3 - X2 X3 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X1,X2,X3)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (!(i1.isvec() && i2.isvec() && i3.isvec()) && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i2.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X1) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (X2) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (X3) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X1) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (X2) must be a matrix" << endl; return 1; }
if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (X3) must be a matrix" << endl; return 1; }
if (dim==0 && (i1.C!=i2.C || i1.C!=i3.C)) { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have same ncols for dim=0" << endl; return 1; }
if (dim==1 && (i1.R!=i2.R || i1.R!=i3.R)) { cerr << progstr+": " << __LINE__ << errstr << "all inputs must have same nrows for dim=1" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0) ? i1.R+i2.R+i3.R : i1.R;
o1.C = (dim==1) ? i1.C+i2.C+i3.C : i1.C;
o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1)
{
    float *X1, *X2, *X3, *Y;
    try { X1 = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
    try { X2 = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
    try { X3 = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (X3)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(X3),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (X3)" << endl; return 1; }
    if (openn::join3_s(Y,X1,X2,X3,int(i1.R),int(i2.R),int(i3.R),int(i1.C),int(i2.C),int(i3.C),dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] X3; delete[] Y;
}
else if (i1.T==101)
{
    float *X1, *X2, *X3, *Y;
    try { X1 = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
    try { X2 = new float[2u*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
    try { X3 = new float[2u*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (X3)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(X3),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (X3)" << endl; return 1; }
    if (openn::join3_c(Y,X1,X2,X3,int(i1.R),int(i2.R),int(i3.R),int(i1.C),int(i2.C),int(i3.C),dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] X3; delete[] Y;
}

//Finish

