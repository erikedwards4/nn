//Includes
#include "bilinear.c"

//Declarations
const valarray<size_t> oktypes = {1u,2u,101u,102u};
const size_t I = 4u, O = 1u;
size_t Ni1, Ni2, No, L;

//Description
string descr;
descr += "IN method.\n";
descr += "Bilinear transformation of 2 layers of inputs.\n";
descr += "\n";
descr += "Input X1 has Ni1 neurons and input X2 has Ni2 neurons\n";
descr += "The output Y has No neurons.\n";
descr += "Each output neuron has a bias term, so B is a vector of length No.\n";
descr += "The weights (W) are a 3D tensor.\n";
descr += "\n";
descr += "If col-major: Y[n,l] = X1[:,l]'*W[:,:,n]*X2[:,l] + B[n] \n";
descr += "where:\n";
descr += "X1 has size Ni1 x L \n";
descr += "X2 has size Ni2 x L \n";
descr += "Y  has size No x L \n";
descr += "W  has size Ni1 x Ni2 x No \n";
descr += "B  has size No x 1 \n";
descr += "\n";
descr += "If row-major: Y[l,n] = X1[l,:]*W[n,:,:]*X2[l,:]' + B[n] \n";
descr += "where:\n";
descr += "X1 has size L x Ni1 \n";
descr += "X2 has size L x Ni2 \n";
descr += "Y  has size L x No \n";
descr += "W  has size No x Ni2 x Ni1 \n";
descr += "B  has size 1 x No \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ bilinear X1 X2 W B -o Y \n";
descr += "$ bilinear X1 X2 W B > Y \n";
descr += "$ cat X | bilinear - W B > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X1,X2,W,B)");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X1) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (X2) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (W) found to be empty" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (B) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (X1) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (X2) must be a matrix" << endl; return 1; }
if (!i3.iscube()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (W) must be a 3D tensor" << endl; return 1; }
if (!i4.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (B) must be a vector" << endl; return 1; }
L = i1.iscolmajor() ? i1.C : i1.R;
Ni1 = i3.iscolmajor() ? i3.R : i3.S;
Ni2 = i3.C;
No = i3.iscolmajor() ? i3.S : i3.R;
if (i4.N()!=No) { cerr << progstr+": " << __LINE__ << errstr << "length of input 4 (B) must equal No (num output neurons)" << endl; return 1; }
if (i1.iscolmajor())
{
    if (i1.R!=Ni1) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X1) must have size Ni1 x L for col-major" << endl; return 1; }
    if (i2.R!=Ni2) { cerr << progstr+": " << __LINE__ << errstr << "Input 2 (X2) must have size Ni2 x L for col-major" << endl; return 1; }
}
else
{
    if (i1.C!=Ni1) { cerr << progstr+": " << __LINE__ << errstr << "Input 1 (X1) must have size L x Ni1 for row-major" << endl; return 1; }
    if (i2.C!=Ni2) { cerr << progstr+": " << __LINE__ << errstr << "Input 2 (X2) must have size L x Ni2 for row-major" << endl; return 1; }
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
    float *X1, *X2, *W, *B, *Y;
    try { X1 = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
    try { X2 = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
    try { W = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (W)" << endl; return 1; }
    try { B = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (B)" << endl; return 1; }
    try { Y = new float[o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(W),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (W)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(B),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (B)" << endl; return 1; }
    if (codee::bilinear_s(Y,X1,X2,W,B,Ni1,Ni2,No,L))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] W; delete[] B; delete[] Y;
}
else if (i1.T==101u)
{
    float *X1, *X2, *W, *B, *Y;
    try { X1 = new float[2u*i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
    try { X2 = new float[2u*i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
    try { W = new float[2u*i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (W)" << endl; return 1; }
    try { B = new float[2u*i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (B)" << endl; return 1; }
    try { Y = new float[2u*o1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(W),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (W)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(B),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (B)" << endl; return 1; }
    if (codee::bilinear_c(Y,X1,X2,W,B,Ni1,Ni2,No,L))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] X1; delete[] X2; delete[] W; delete[] B; delete[] Y;
}

//Finish
