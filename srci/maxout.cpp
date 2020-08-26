//Includes
#include "maxout.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
size_t dim, M;

//Description
string descr;
descr += "Layer activation function.\n";
descr += "Gets maxout function for each vector in X.\n";
descr += "This is not just the max along vectors of X.\n";
descr += "Rather, each vector in X has M groups of No values,\n";
descr += "as derived from M separate affine transforms,\n";
descr += "and stacked into vectors of length Lx = M*No.\n";
descr += "Each of the No neurons takes the max over M inputs.\n";
descr += "\n";
descr += "Use -m (--M) to give the number of input groups.\n";
descr += "\n";
descr += "Use -d (--dim) to specify the axis of the vectors within X.\n";
descr += "This is the dimension of length No, \n";
descr += "where No is the number of outputs from the layer.\n";
descr += "The default is 0 (along cols) unless X is a vector.\n";
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
if (a_d->count==0) { dim = i1.isvec() ? i1.nonsingleton1() : 0u; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = size_t(a_d->ival[0]); }
if (dim>3u) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1,2,3}" << endl; return 1; }

//Get M
if (a_m->count==0) { M = 1u; }
else if (a_m->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "M must be positive" << endl; return 1; }
else { M = size_t(a_m->ival[0]); }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) must be a matrix" << endl; return 1; }
if (dim==0u && i1.R%M) { cerr << progstr+": " << __LINE__ << errstr << "num rows X must be a multiple of M for dim=0" << endl; return 1; }
if (dim==1u && i1.C%M) { cerr << progstr+": " << __LINE__ << errstr << "num cols X must be a multiple of M for dim=1" << endl; return 1; }
if (dim==2u && i1.S%M) { cerr << progstr+": " << __LINE__ << errstr << "num slices X must be a multiple of M for dim=2" << endl; return 1; }
if (dim==3u && i1.H%M) { cerr << progstr+": " << __LINE__ << errstr << "num hyperslices X must be a multiple of M for dim=3" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = (dim==0u) ? i1.R/M : i1.R;
o1.C = (dim==1u) ? i1.C/M : i1.C;
o1.S = (dim==2u) ? i1.S/M : i1.S;
o1.H = (dim==3u) ? i1.H/M : i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::maxout_s(Y,X,i1.R,i1.C,i1.S,i1.H,i1.iscolmajor(),dim,M))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}

//Finish
