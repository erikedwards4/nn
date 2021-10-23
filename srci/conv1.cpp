//Includes
#include "conv1.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 3u, O = 1u;
size_t Nb, Ni, No, Li, Lo, Lk, str;
int pad, Ti, ceil_mode, pm;
string pad_mode;

//Description
string descr;
descr += "1d convolution, same as conv1d, but no dilation.\n";
descr += "\n";
descr += "This is an input (IN) component for a layer of neurons.\n";
descr += "There are No (num output) neurons in the layer,\n";
descr += "and the layer gets inputs from Ni (num input) neurons.\n";
descr += "\n";
descr += "X has size Ni x Li x Nb for col-major.\n";
descr += "X has size Nb x Li x Ni for row-major.\n";
descr += "where:\n";
descr += "Ni is the num input neurons (aka, C_in or in_channels).\n";
descr += "Li is the input length (usually the number of time points).\n";
descr += "Nb is the batch size (if 1, can use matrices, see below).\n";
descr += "\n";
descr += "K is the convolving kernel (weights).\n";
descr += "K has size Ni x Lk x No for col-major.\n";
descr += "K has size No x Lk x Ni for row-major.\n";
descr += "where:\n";
descr += "No is the num output neurons (aka, C_out or out_channels).\n";
descr += "Lk is the kernel length (kernel_size, or time width).\n";
descr += "\n";
descr += "B is a vector of length No (each output neuron has a fixed bias).\n";
descr += "\n";
descr += "Y has size No x Lo x Nb for col-major.\n";
descr += "Y has size Nb x Lo x No for row-major.\n";
descr += "where:\n";
descr += "Lo =  ceil[1 + (Li + 2*pad - Lk)/stride], for ceil_mode true.\n";
descr += "Lo = floor[1 + (Li + 2*pad - Lk)/stride], for ceil_mode false.\n";
descr += "\n";
descr += "If Nb==1:\n";
descr += "X is a matrix with size Ni x Li (col-major) or Li x Ni (row-major);\n";
descr += "Y is a matrix with size No x Lo (col-major) or Lo x No (row-major).\n";
descr += "\n";
descr += "Include -c (--ceil_mode) to use ceil for Lo calculation [default=false].\n";
descr += "\n";
descr += "Use -s (--stride) to give the stride (step-size) in samples [default=1].\n";
descr += "\n";
descr += "Use -p (--padding) to give the padding length in samples [default=0]\n";
descr += "\n";
descr += "Use -m (--pad_mode) to give the padding mode as [default='zeros']\n";
descr += "The pad_mode can be 'zeros', 'reflect', 'repeat' or 'circular'.\n";
descr += "The pad_mode can also be entered as 'z', 'ref', 'rep' or 'c'.\n";
descr += "\n";
descr += "Examples:\n";
descr += "$ conv1 X K B -o Y \n";
descr += "$ conv1 -s2 -p10 -m'reflect' X K B > Y \n";
descr += "$ cat X | conv1 -c -s5 -p10 -m'c' - K B > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,K,B)");
struct arg_int  *a_str = arg_intn("s","stride","<uint>",0,1,"stride in samps [default=1]");
struct arg_int  *a_pad = arg_intn("p","padding","<int>",0,1,"padding [default=0]");
struct arg_str   *a_pm = arg_strn("m","pad_mode","<str>",0,1,"padding mode [default='zeros']");
struct arg_lit   *a_cm = arg_litn("c","ceil_mode",0,1,"include to use ceil_mode [default=false]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get stride
if (a_str->count==0) { str = 1u; }
else if (a_str->ival[0]<1) { cerr << progstr+": " << __LINE__ << errstr << "stride must be positive" << endl; return 1; }
else { str = size_t(a_str->ival[0]); }

//Get ceil_mode
ceil_mode = (a_cm->count>0);

//Get padding
if (a_pad->count==0) { pad = 0; }
else { pad = a_pad->ival[0]; }

//Get pad_mode and pm
if (a_pm->count==0) { pad_mode = "zeros"; }
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
else
{
    cerr << progstr+": " << __LINE__ << errstr << "pad_mode must be in {'z','rep','ref','c'} or {'zeros','replicate','reflect','circular'}" << endl;
    return 1;
}

//Checks
if (i1.ismat())
{
    Ni = i1.iscolmajor() ? i1.R :i1.C;
    Li = i1.iscolmajor() ? i1.C : i1.R;
    Nb = 1u;
}
else
{
    Ni = i1.iscolmajor() ? i1.R : i1.S;
    Li = i1.iscolmajor() ? i1.C : i1.C;
    Nb = i1.iscolmajor() ? i1.S : i1.R;
}
No = i2.iscolmajor() ? i2.S : i2.R;
Lk = i2.C;
Ti = (int)Li + 2*pad;
if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (K) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) found to be empty" << endl; return 1; }
if (!i1.iscube()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X) must be a cube (3D tensor)" << endl; return 1; }
if (!i2.iscube()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (K) must be a cube (3D tensor)" << endl; return 1; }
if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) must be a vector" << endl; return 1; }
if (i3.N()!=No) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (B) must have length No (num output neurons)" << endl; return 1; }
if (Lk>Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= Lk" << endl; return 1; }
if (pad<=-(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "pad length must be > -Li" << endl; return 1; }
if (pm && pad>(int)Li) { cerr << progstr+": " << __LINE__ << errstr << "Li (length of input vecs) must be >= pad length" << endl; return 1; }
if (Ti<(int)Lk) { cerr << progstr+": " << __LINE__ << errstr << "Li+2*pad must be >= Lk" << endl; return 1; }

//Set output header info
Lo = 1u + size_t(Ti-(int)Lk)/str + size_t(ceil_mode && size_t(Ti-(int)Lk)%str);
o1.F = i1.F; o1.T = i1.T;
if (i1.ismat())
{
    o1.R = i1.iscolmajor() ? No : Lo;
    o1.C = i1.iscolmajor() ? Lo : No;
    o1.S = i1.S;
}
else
{
    o1.R = i1.iscolmajor() ? No : Nb;
    o1.C = i1.iscolmajor() ? Lo : Lo;
    o1.S = i1.iscolmajor() ? Nb : No;
}
o1.H = i1.H;

//Other prep

//Process
if (i1.T==1u)
{
    float *X, *K, *B, *Y;
    try { X = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { K = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (K)" << endl; return 1; }
    try { B = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(K),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (K)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(B),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (codee::conv1_s(Y,X,K,B,Ni,No,Nb,Li,Lk,pad,str,ceil_mode,pm))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] K; delete[] B; delete[] Y;
}
else if (i1.T==101u)
{
    float *X, *K, *B, *Y;
    try { X = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
    try { K = new float[2u*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (K)" << endl; return 1; }
    try { B = new float[2u*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(K),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (K)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(B),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
    if (codee::conv1_c(Y,X,K,B,Ni,No,Nb,Li,Lk,pad,str,ceil_mode,pm))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X; delete[] K; delete[] B; delete[] Y;
}

//Finish
