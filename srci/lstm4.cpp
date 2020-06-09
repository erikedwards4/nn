//Includes
//#include <chrono>
#include "lstm4.c"

//Declarations
const valarray<uint8_t> oktypes = {1,2};
const size_t I = 8, O = 1;
int dim, N, T;

//Description
string descr;
descr += "Neural soma stage.\n";
descr += "Does LSTM (long short-term memory) model for driving\n";
descr += "inputs Xc, Xi, Xf, Xo, where Xc is the input to the cell,\n";
descr += "and Xi, Xf, Xo are the inputs for the input, forget, output gates.\n";
descr += "\n";
descr += "All have size NxT or TxN, where N is the number of neurons,\n";
descr += "and T is the number of observations (e.g. time points).\n";
descr += "\n";
descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
descr += "\n";
descr += "For dim=0, C[:,t] = tanh{Xc[:,t] + Uc*Y[:,t-1]} \n";
descr += "           I[:,t] = sig{Xi[:,t] + Ui*Y[:,t-1]} \n";
descr += "           F[:,t] = sig{Xf[:,t] + Uf*Y[:,t-1]} \n";
descr += "           O[:,t] = sig{Xo[:,t] + Uo*Y[:,t-1]} \n";
descr += "           H[:,t] = F[:,t].*H[:,t-1] + I[:,t].*C[:,t] \n";
descr += "           Y[:,t] = O[:,t].*tanh{C[:,t]} \n";
descr += "with sizes Xc, Xi, Xf, Xo: N x T \n";
descr += "           Uc, Ui, Uf, Uo: N x N \n";
descr += "           Y             : N x T \n";
descr += "\n";
descr += "For dim=1, C[t,:] = tanh{Xc[t,:] + Y[t-1,:]*Uc} \n";
descr += "           I[t,:] = sig{Xi[t,:] + Y[t-1,:]*Ui} \n";
descr += "           F[t,:] = sig{Xf[t,:] + Y[t-1,:]*Uf} \n";
descr += "           O[t,:] = sig{Xo[t,:] + Y[t-1,:]*Uo} \n";
descr += "           H[t,:] = F[t,:].*H[t-1,:] + I[t,:].*C[t,:] \n";
descr += "           Y[t,:] = O[t,:].*tanh{C[t,:]} \n";
descr += "with sizes Xc, Xi, Xf, Xo: T x N \n";
descr += "           Uc, Ui, Uf, Uo: N x N \n";
descr += "           Y             : T x N \n";
descr += "\n";
descr += "Examples:\n";
descr += "$ lstm4 Xc Xi Xf Xo Uc Ui Uf Uo -o Y \n";
descr += "$ lstm4 Xc Xi Xf Xo Uc Ui Uf Uo > Y \n";
descr += "$ cat X | lstm4 - Xi Xf Xo Uc Ui Uf Uo > Y \n";

//Argtable
struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (Xc,Xi,Xf,Xo,Uc,Ui,Uf,Uo)");
struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");

//Get options

//Get dim
if (a_d->count==0) { dim = 0; }
else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
else { dim = a_d->ival[0]; }
if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

//Checks
if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T || i1.T!=i5.T || i1.T!=i6.T || i1.T!=i7.T || i1.T!=i8.T)
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same data type" << endl; return 1; }
if (!i1.isvec() && (i1.iscolmajor()!=i2.iscolmajor() || i1.iscolmajor()!=i3.iscolmajor() || i1.iscolmajor()!=i4.iscolmajor() || i1.iscolmajor()!=i5.iscolmajor() || i1.iscolmajor()!=i6.iscolmajor() || i1.iscolmajor()!=i7.iscolmajor() || i1.iscolmajor()!=i8.iscolmajor()))
{ cerr << progstr+": " << __LINE__ << errstr << "all inputs must have the same row/col major format" << endl; return 1; }
if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xc) found to be empty" << endl; return 1; }
if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) found to be empty" << endl; return 1; }
if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xf) found to be empty" << endl; return 1; }
if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Xo) found to be empty" << endl; return 1; }
if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Uc) found to be empty" << endl; return 1; }
if (i6.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Ui) found to be empty" << endl; return 1; }
if (i7.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (Uf) found to be empty" << endl; return 1; }
if (i8.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (Uo) found to be empty" << endl; return 1; }
if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xc) must be a matrix" << endl; return 1; }
if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) must be a matrix" << endl; return 1; }
if (!i3.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (Xf) must be a matrix" << endl; return 1; }
if (!i4.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (Xo) must be a matrix" << endl; return 1; }
if (!i5.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (Uc) must be a matrix" << endl; return 1; }
if (!i6.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (Ui) must be a matrix" << endl; return 1; }
if (!i7.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (Uf) must be a matrix" << endl; return 1; }
if (!i8.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (Uo) must be a matrix" << endl; return 1; }
if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-4 (Xc, Xi, Xf, Xo) must have the same size" << endl; return 1; }
if (i1.R!=i3.R || i1.C!=i3.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-4 (Xc, Xi, Xf, Xo) must have the same size" << endl; return 1; }
if (i1.R!=i4.R || i1.C!=i4.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1-4 (Xc, Xi, Xf, Xo) must have the same size" << endl; return 1; }
if (i5.R!=i6.R || i5.C!=i6.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must have the same size" << endl; return 1; }
if (i5.R!=i7.R || i5.C!=i7.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must have the same size" << endl; return 1; }
if (i5.R!=i8.R || i5.C!=i8.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must have the same size" << endl; return 1; }
if (i5.R!=i5.C || i6.R!=i6.C || i7.R!=i7.C || i8.R!=i8.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must be square" << endl; return 1; }
if (dim==0 && i1.R!=i5.R) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must have size NxN" << endl; return 1; }
if (dim==1 && i1.C!=i5.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 5-8 (Uc, Ui, Uf, Uo) must have size NxN" << endl; return 1; }

//Set output header info
o1.F = i1.F; o1.T = i1.T;
o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;

//Other prep
N = (dim==0) ? int(o1.R) : int(o1.C);
T = (dim==0) ? int(o1.C) : int(o1.R);

//Process
if (i1.T==1)
{
    float *Xc, *Xi, *Xf, *Xo, *Uc, *Ui, *Uf, *Uo;// *Y;
    try { Xc = new float[i1.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (Xc)" << endl; return 1; }
    try { Xi = new float[i2.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xi)" << endl; return 1; }
    try { Xf = new float[i3.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (Xf)" << endl; return 1; }
    try { Xo = new float[i4.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (Xo)" << endl; return 1; }
    try { Uc = new float[i5.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (Uc)" << endl; return 1; }
    try { Ui = new float[i6.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (Ui)" << endl; return 1; }
    try { Uf = new float[i7.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 7 (Uf)" << endl; return 1; }
    try { Uo = new float[i8.N()]; }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 8 (Uo)" << endl; return 1; }
    //try { Y = new float[o1.N()]; }
    //catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
    try { ifs1.read(reinterpret_cast<char*>(Xc),i1.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (Xc)" << endl; return 1; }
    try { ifs2.read(reinterpret_cast<char*>(Xi),i2.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xi)" << endl; return 1; }
    try { ifs3.read(reinterpret_cast<char*>(Xf),i3.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (Xf)" << endl; return 1; }
    try { ifs4.read(reinterpret_cast<char*>(Xo),i4.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (Xo)" << endl; return 1; }
    try { ifs5.read(reinterpret_cast<char*>(Uc),i5.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (Uc)" << endl; return 1; }
    try { ifs6.read(reinterpret_cast<char*>(Ui),i6.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (Ui)" << endl; return 1; }
    try { ifs7.read(reinterpret_cast<char*>(Uf),i7.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 7 (Uf)" << endl; return 1; }
    try { ifs8.read(reinterpret_cast<char*>(Uo),i8.nbytes()); }
    catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 8 (Uo)" << endl; return 1; }
    //auto tic = chrono::high_resolution_clock::now();
    //if (openn::lstm4_s(Y,Xc,Xi,Xf,Xo,Uc,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
    if (openn::lstm4_inplace_s(Xc,Xi,Xf,Xo,Uc,Ui,Uf,Uo,N,T,dim,i1.iscolmajor()))
    { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
    if (wo1)
    {
        try { ofs1.write(reinterpret_cast<char*>(Xc),o1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
    }
    delete[] Xc; delete[] Xi; delete[] Xf; delete[] Xo; delete[] Uc; delete[] Ui; delete[] Uf; delete[] Uo; //delete[] Y;
    //auto toc = chrono::high_resolution_clock::now();
    //auto dur = chrono::duration_cast<chrono::microseconds>(toc-tic);
    //cerr << dur.count()/1000.0 << " ms" << endl; 
}

//Finish

