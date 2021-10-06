//Includes
#include "avgpool1d.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 1u, O = 1u;
size_t N, Li, Lo, Lk, str, dil;
int pad, Ti, Tk, ceil_mode, pm;
string pad_mode;

//Description
string descr;
descr += "1d average-pooling, with full opts of conv1d (e.g., dilation).\n";
descr += "\n";
descr += "This is an input (IN) component for a layer of neurons.\n";
descr += "There are N neurons in the layer,\n";
descr += "and the layer gets inputs from N neurons.\n";
descr += "\n";
descr += "X has size N x Li for col-major.\n";
descr += "X has size Li x N for row-major.\n";
descr += "where Li is the input length (usually the number of time points).\n";
descr += "\n";
descr += "Lk is the kernel length (kernel_size, or time width).\n";
descr += "There is no explicit kernel; this is by analogy to conv1.\n";
descr += "Thus, Lk is the num samps included in each average.\n";
descr += "\n";
descr += "Y has size N x Lo for col-major.\n";
descr += "Y has size Lo x N for row-major.\n";
descr += "where:\n";
descr += "Lo =  ceil[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], for ceil_mode true.\n";
descr += "Lo = floor[1 + (Li + 2*pad - dil*(Lk-1) - 1)/stride], for ceil_mode false.\n";
descr += "\n";
descr += "Include -c (--ceil_mode) to use ceil for Lo calculation [default=false].\n";
descr += "\n";
descr += "Use -s (--stride) to give the stride (step-size) in samples [default=1].\n";
descr += "\n";
descr += "Use -i (--dilation) to give the dilation factor [default=1].\n";
descr += "\n";
descr += "Use -p (--padding) to give the padding length in samples [default=0]\n";
descr += "\n";
descr += "Use -m (--pad_mode) to give the padding mode as [default='n']\n";
descr += "The pad_mode can be 'zeros', 'reflect', 'repeat', 'circular' or 'no_count_pad'.\n";
descr += "The pad_mode can also be entered as 'z', 'ref', 'rep', 'c' or 'n'.\n";
descr += "Use pad_mode 'no_count_pad' or 'n' to emulate count_include_pad=False.\n";
descr += "Use pad_mode 'no_count_pad' or 'n' to emulate usual AvgPool1d behavior.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ avgpool1d -k5 X -o Y \n";
descr += "$ avgpool1d -k9 -s2 -p10 -m'reflect' X > Y \n";
descr += "$ cat X | avgpool1d -k12 -i2 -c -s5 -p10 -m'c' - > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input file (X)");
struct arg_int   *a_lk = arg_intn("k","kernel_size","<uint>",0,1,"kernel length in samps [default=1]");
struct arg_int  *a_str = arg_intn("s","step","<uint>",0,1,"step size in samps [default=1]");
struct arg_int  *a_dil = arg_intn("i","dilation","<uint>",0,1,"dilation factor [default=1]");
struct arg_int  *a_pad = arg_intn("p","padding","<int>",0,1,"padding [default=0]");
struct arg_str   *a_pm = arg_strn("m","pad_mode","<str>",0,1,"padding mode [default='zeros']");
struct arg_lit   *a_cm = arg_litn("c","ceil_mode",0,1,"include to use ceil_mode [default=false]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get Lk
if (a_lk->count==0) { Lk = 1u; }
else if (a_lk->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "kernel_size must be positive" << endl; return 1; }
else { Lk = size_t(a_lk->ival[0]); }

//Get stride
if (a_str->count==0) { str = 1u; }
else if (a_str->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "stride must be positive" << endl; return 1; }
else { str = size_t(a_str->ival[0]); }

//Get dil
if (a_dil->count==0) { dil = 1u; }
else if (a_dil->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "dilation must be positive" << endl; return 1; }
else { dil = size_t(a_dil->ival[0]); }

//Get ceil_mode
ceil_mode = (a_cm->count>0);

//Get padding
if (a_pad->count==0) { pad = 0; }
else { pad = a_pad->ival[0]; }

//Get pad_mode and pm
if (a_pm->count==0) { pad_mode = "n"; }
else
{
	try { pad_mode = string(a_pm->sval[0]); }
	catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem getting string for pad_mode" << endl; return 1; }
}
for (string::size_type c=0u; c<pad_mode.size(); ++c) { pad_mode[c] = char(tolower(pad_mode[c])); }
if (pad_mode=="z" || pad_mode=="zeros") { pm = 0; }
else if (pad_mode=="rep" || pad_mode=="replicate") { pm = 1; }
else if (pad_mode=="ref" || pad_mode=="reflect") { pm = 2; }
else if (pad_mode=="c" || pad_mode=="circular") { pm = 3; }
else if (pad_mode=="n" || pad_mode=="no_count_pad") { pm = 4; }
else
{
    cerr << progstr+": " << __LINE__ << errstr << "pad_mode must be in {'z','rep','ref','c','n'} or {'zeros','replicate','reflect','circular','no_count_pad'}" << endl;
    return 1;
}

//Checks
N = i1.iscolmajor() ? i1.R : i1.C;
Li = i1.iscolmajor() ? i1.C : i1.R;
Ti = (int)Li + 2*pad;
Tk = (int)(dil*(Lk-1u)) + 1;
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (Lk>Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= Lk" << endl; return 1; }
if (pad<=-(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "pad length must be > -Li" << endl; return 1; }
if (pm && pad>(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= pad length" << endl; return 1; }
if (Ti<(int)Lk) { cerr << progstr+": " << __LINE__ << errstr << "Li+2*pad must be >= Lk" << endl; return 1; }
if (Ti<Tk) { cerr << progstr+": " << __LINE__ << errstr << "Li+2*pad must be >= dil*(Lk-1)" << endl; return 1; }

//Set output header info
Lo = 1u + size_t(Ti-Tk)/str + size_t(ceil_mode && size_t(Ti-Tk)%str);
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.iscolmajor() ? N : Lo;
o1.C = i1.iscolmajor() ? Lo : N;
o1.S = i1.S; o1.H = i1.H;

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
    if (codee::avgpool1d_s(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}
else if (i1.T==101u)
{
    float *X, *Y;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file (X)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file (X)" << endl; return 1; }
    if (codee::avgpool1d_c(Y,X,N,Li,Lk,pad,str,dil,ceil_mode,pm))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] Y;
}

//Finish
