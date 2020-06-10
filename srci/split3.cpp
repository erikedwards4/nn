//Includes
#include "split3.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2,101,102};
const size_t I = 1, O = 3;
int dim;

//Description
string descr;
descr += "Splits 1 input X into 3 equal-sized outputs Y1, Y2, Y3.\n";
descr += "\n";
descr += "Use -d (--dim) to give the dimension (axis) [default=0].\n";
descr += "Use -d0 to work along cols --> Y has size R/3 x C.\n";
descr += "Use -d1 to work along rows --> Y has size R x C/3.\n";
descr += "\n";
descr += "For dim=0, num rows X must be even (R%3==0).\n";
descr += "For dim=1, num cols X must be even (C%3==0).\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ split3 X -o Y1 -o Y2 -o Y3 \n";
descr += "$ split3 X -o Y1 -o Y2 > Y3 \n";
descr += "$ split3 -d1 X -o Y1 -o Y2 -o Y3 \n";
descr += "$ cat X | split3 -o Y1 -o Y2 -o Y3 \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output files (Y1,Y2,Y3)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
if (dim==0 && i1.R%3) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be a multiple of 3 for dim=0" << endl; return 1; }
if (dim==1 && i1.C%3) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be a multiple of 3 for dim=1" << endl; return 1; }

//Set output header info
o1.F = o2.F = o3.F = i1.F;
o1.T = o2.T = o3.F = i1.T;
o1.R = o2.R = o3.R = (dim==0) ? i1.R/3 : i1.R;
o1.C = o2.C = o3.C = (dim==1) ? i1.C/3 : i1.C;
o1.S = o2.S = o3.S = i1.S;
o1.H = o2.H = o3.H = i1.H;

//Other prep

//Process
if (i1.T==1)
{
    float *X, *Y1, *Y2, *Y3;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y1 = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
    try { Y2 = new float[o2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
    try { Y3 = new float[o3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 3 (Y3)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (openn::split3_s(Y1,Y2,Y3,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
        try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        try { ofs3.write(reinterpret_cast<char*>(Y3),o3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 3 (Y3)" << endl; return 1; }
    }
    delete[] X; delete[] Y1; delete[] Y2; delete[] Y3;
}
else if (i1.T==101)
{
    float *X, *Y1, *Y2, *Y3;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y1 = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 1 (Y1)" << endl; return 1; }
    try { Y2 = new float[2u*o2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 2 (Y2)" << endl; return 1; }
    try { Y3 = new float[2u*o3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file 3 (Y3)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (openn::split3_c(Y1,Y2,Y3,X,int(i1.R),int(i1.C),dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y1),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 1 (Y1)" << endl; return 1; }
        try { ofs2.write(reinterpret_cast<char*>(Y2),o2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 2 (Y2)" << endl; return 1; }
        try { ofs3.write(reinterpret_cast<char*>(Y3),o3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file 3 (Y3)" << endl; return 1; }
    }
    delete[] X; delete[] Y1; delete[] Y2; delete[] Y3;
}

//Finish

