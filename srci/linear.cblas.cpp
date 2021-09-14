//Includes
#include "linear.cblas.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 2u, O = 1u;
size_t Ni, No, L;

//Description
string descr;
descr += "IN method.\n";
descr += "Linear transformation (weights only) of Ni inputs to No outputs.\n";
descr += "\n";
descr += "Input X has Ni neurons and output Y has No neurons.\n";
descr += "\n";
descr += "The Ni or No neurons are always contiguous in memory, such that:\n";
descr += "\n";
descr += "If col-major: Y[:,l] = W' * X[:,l] \n";
descr += "where:\n";
descr += "X has size Ni x L \n";
descr += "Y has size No x L \n";
descr += "W has size Ni x No \n";
descr += "\n";
descr += "If row-major: Y[l,:] = X[l,:] * W' \n";
descr += "where:\n";
descr += "X has size L x Ni \n";
descr += "Y has size L x No \n";
descr += "W has size No x Ni \n";
descr += "\n";
descr += "Note that W is transposed (non-conjugate), \n";
descr += "such that vecs of length Ni are contiguous in memory.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ linear.cblas X W -o Y \n";
descr += "$ linear.cblas X W > Y \n";
descr += "$ cat X | linear.cblas - W > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,W)");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Checks
if (i1.T!=i2.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (W) must be a matrix" << endl; return 1; }
L = i1.iscolmajor() ? i1.C : i1.R;
Ni = i2.iscolmajor() ? i2.R : i2.C;
No = i2.iscolmajor() ? i2.C : i2.R;
if (i1.iscolmajor())
{
    if (i1.R!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X) must have size Ni x L for col-major" << endl; return 1; }
}
else
{
    if (i1.C!=Ni) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X) must have size L x Ni for row-major" << endl; return 1; }
}

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.iscolmajor() ? No : L;
o1.C = i1.isrowmajor() ? No : L;
o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X, *W, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    if (codee::linear_cblas_s(Y,X,W,Ni,No,L))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] W; delete[] Y;
}
else if (i1.T==101u)
{
    float *X, *W, *Y;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { W = new float[2u*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (W)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(W),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (W)" << endl; return 1; }
    if (codee::linear_cblas_c(Y,X,W,Ni,No,L))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] W; delete[] Y;
}

//Finish
