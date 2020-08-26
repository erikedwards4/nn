//Includes
#include "affine.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 3u, O = 1u;
size_t dim, Ni, No;

//Description
string descr;
descr += "Linear algebra function for 3 inputs.\n";
descr += "Also a vec2vec operation upon input 1 (X).\n";
descr += "Affine transformation of each vec in X using matrix W and vec B.\n";
descr += "X can be a tensor up to 4D, with vecs along any dimension.\n";
descr += "\n";
descr += "Use -d (--dim) to give the dimension (axis) [default=0].\n";
descr += "Use -d0 to transform each col vec.\n";
descr += "Use -d1 to transform each row vec.\n";
descr += "Use -d2 to transform each tube vec.\n";
descr += "Use -d3 to transform each hypertube vec.\n";
descr += "\n";
descr += "Y has same size as X, except along dim it has length No \n";
descr += "(the length of B), which equals the number of rows in W.\n";
descr += "\n";
descr += "Each vector in X has length Ni, and each vector in Y has length No.\n";
descr += "Vector B has length No. \n";
descr += "This assumes that W has leading dimension Ni! \n";
descr += "If colmajor, then W has size Ni x No. \n";
descr += "If rowmajor, then W has size No x Ni. \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ affine X W B -o Y \n";
descr += "$ affine X W B > Y \n";
descr += "$ cat X | affine - W B > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,W,B)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = i1.isvec() ? i1.nonsingleton1() : 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>3u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1,2,3}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) found to be empty" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) must be a matrix" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) must be a vector" << endl; return 1; }
if (!major_compat(i1,i2)) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1 and 2 must have the same row/col major format" << endl; return 1; }
Ni = (i1.iscolmajor()) ? i2.R : i2.C;
No = (i1.iscolmajor()) ? i2.C : i2.R;
if (No!=i3.N()) { cerr << progstr+": " << __LINE__ << errstr << "length of input 3 (B) must equal nrows of input 2 (W)" << endl; return 1; }
if (dim==0u && i1.R!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "length of vecs in input 1 (X) must equal Ni of input 2 (W)" << endl; return 1; }
if (dim==1u && i1.C!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "length of vecs in input 1 (X) must equal Ni of input 2 (W)" << endl; return 1; }
if (dim==2u && i1.S!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "length of vecs in input 1 (X) must equal Ni of input 2 (W)" << endl; return 1; }
if (dim==3u && i1.H!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "length of vecs in input 1 (X) must equal Ni of input 2 (W)" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0u) ? No : i1.R;
o1.C = (dim==1u) ? No : i1.C;
o1.S = (dim==2u) ? No : i1.S;
o1.H = (dim==3u) ? No : i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X1, *X2, *X3, *Y;
    try { X1 = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { X2 = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { X3 = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(X3),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (codee::affine_s(Y,X1,X2,X3,i1.R,i1.C,i1.S,i1.H,No,o1.iscolmajor(),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] X3; delete[] Y;
}
else if (i1.T==101u)
{
    float *X1, *X2, *X3, *Y;
    try { X1 = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { X2 = new float[2u*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { X3 = new float[2u*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(X3),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (codee::affine_c(Y,X1,X2,X3,i1.R,i1.C,i1.S,i1.H,No,o1.iscolmajor(),dim))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] X3; delete[] Y;
}

//Finish
