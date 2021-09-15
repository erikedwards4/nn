//Includes
#include "rrelu.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u};
const size_t I = 1u, O = 1u;
double lower, upper;

//Description
string descr;
descr += "Activation function.\n";
descr += "Gets randomized ReLU (RReLU) of each element of X [Xu et al. 2015].\n";
descr += "For each element: y = alpha*x,  if x<0. \n";
descr += "                  y = x,        if x>=0. \n";
descr += "\n";
descr += "where alpha is drawn from a uniform distribution in [lower upper],\n";
descr += "and 0 <= lower < upper < 1.\n\n";
descr += "\n";
descr += "Use -l (--lower) to specify the lower end of the range for alpha [default=0.125].\n";
descr += "Use -u (--upper) to specify the upper end of the range for alpha [default=1/3].\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ rrelu X -o Y \n";
descr += "$ rrelu -l0.1 -u0.5 X > Y \n";
descr += "$ cat X | rrelu -u0.25 > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_dbl    *a_l = arg_dbln("l","lower","<dbl>",0,1,"lower end of range [default=0.125]");
struct arg_dbl    *a_u = arg_dbln("u","upper","<dbl>",0,1,"upper end of range [default=1/3]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get lower
lower = (a_l->count==0) ? 0.125 : a_l->dval[0];
if (lower<0.0 || lower>=1.0) { cerr << progstr+": " << __LINE__ << errstr << "lower must be in [0 1)" << endl; return 1; }

//Get val
upper = (a_u->count==0) ? 1.0/3.0 : a_u->dval[0];
if (upper<0.0 || upper>=1.0) { cerr << progstr+": " << __LINE__ << errstr << "upper must be in [0 1)" << endl; return 1; }
if (upper<=lower) { cerr << progstr+": " << __LINE__ << errstr << "upper must be > lower" << endl; return 1; }

//Checks
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input (X) found to be empty" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::rrelu_inplace_s(X,i1.N(),float(lower),float(upper)))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(X),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X;
}

//Finish
