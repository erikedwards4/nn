//@author Erik Edwards
//@date 2019-2020


#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <complex>
#include <unordered_map>
#include <argtable2.h>
#include "/home/erik/codee/cmli/cmli.hpp"
#include "grossberg2.c"

#ifdef I
#undef I
#endif


int main(int argc, char *argv[])
{
    using namespace std;


    //Declarations
    int ret = 0;
    const string errstr = ": \033[1;31merror:\033[0m ";
    const string warstr = ": \033[1;35mwarning:\033[0m ";
    const string progstr(__FILE__,string(__FILE__).find_last_of("/")+1,strlen(__FILE__)-string(__FILE__).find_last_of("/")-5);
    const valarray<uint8_t> oktypes = {1,2};
    const size_t I = 8, O = 1;
    ifstream ifs1, ifs2, ifs3, ifs4, ifs5, ifs6, ifs7, ifs8; ofstream ofs1;
    int8_t stdi1, stdi2, stdi3, stdi4, stdi5, stdi6, stdi7, stdi8, stdo1, wo1;
    ioinfo i1, i2, i3, i4, i5, i6, i7, i8, o1;
    int dim, n, N, T;
    double fs;


    //Description
    string descr;
    descr += "Neural soma stage.\n";
    descr += "Does middle stage of Grossberg model for each row or col of Xe and Xi.\n";
    descr += "which are separate excitatory and inhibitory driving inputs.\n";
    descr += "\n";
    descr += "The soma model is temporal integration (1st-order IIR filter, time-constant tau),\n";
    descr += "along with a subtractive feedback with gain alpha,\n";
    descr += "along with adaptive gain control with parameters beta and gamma.\n";
    descr += "\n";
    descr += "Xe and Xi have size NxT or TxN, where N is the number of neurons\n";
    descr += "and T is the number of observations (e.g. time points).\n";
    descr += "\n";
    descr += "Use -d (--dim) to specify the dimension (axis) of length N [default=0].\n";
    descr += "\n";
    descr += "For dim=0, Y[n,t] = a[n]*Y[n,t-1] + b[n]*(g[n,t]*X[n,t]-alpha[n]*Y[n,t-1]). \n";
    descr += "with sizes X:  N x T \n";
    descr += "           Y:  N x T \n";
    descr += "\n";
    descr += "For dim=1, Y[t,n] = a[n]*Y[t-1,n] + b[n]*(g[t,n]*X[t,n]-alpha[n]*Y[t-1,n]). \n";
    descr += "with sizes X:  T x N \n";
    descr += "           Y:  T x N \n";
    descr += "\n";
    descr += "where a[n] = exp(-1/(fs*taus[n])) and b[n] = 1 - a[n].\n";
    descr += "and g[n,t] = gamma[n] - beta[n]*Y[n,t-1].\n";
    descr += "\n";
    descr += "Use -s (--fs) to give the sample rate of X in Hz [default=10000].\n";
    descr += "\n";
    descr += "Enter a vector of N taus as the 3rd input.\n";
    descr += "Or enter a single tau as the 3rd input, to be used by all N neurons.\n";
    descr += "\n";
    descr += "Enter a vector of N alphas as the 4th input.\n";
    descr += "Or enter a single alpha as the 4th input, to be used by all N neurons.\n";
    descr += "\n";
    descr += "Enter vectors of N betas as the 5th and 6th inputs (e and i separately).\n";
    descr += "Or enter a betae and betai as the 5th and 6th inputs, to be used by all N neurons.\n";
    descr += "\n";
    descr += "Enter vector of N gammas as the 7th and 8th inputs (e and i separately).\n";
    descr += "Or enter a gammae and gammai as the 7th and 8th inputs, to be used by all N neurons.\n";
    descr += "\n";
    descr += "Examples:\n";
    descr += "$ grossberg2 Xe Xi tau alpha betae betai gammae gammai -o Y \n";
    descr += "$ grossberg2 Xe Xi tau alpha betae betai gammae gammai > Y \n";
    descr += "$ cat X | grossberg2 - tau alpha betae betai gammae gammai > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (Xe,Xi,tau,alpha,betae,betai,gammae,gammai)");
    struct arg_int    *a_d = arg_intn("d","dim","<uint>",0,1,"dimension (0 or 1) [default=0]");
    struct arg_dbl   *a_fs = arg_dbln("s","fs","<dbl>",0,1,"sample rate of Xe and Xi [default=10000]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_d, a_fs, a_fo, a_help, a_end};
    if (arg_nullcheck(argtable)!=0) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating argtable" << endl; return 1; }
    nerrs = arg_parse(argc, argv, argtable);
    if (a_help->count>0)
    {
        cout << "Usage: " << progstr; arg_print_syntax(stdout, argtable, "\n");
        cout << endl; arg_print_glossary(stdout, argtable, "  %-25s %s\n");
        cout << endl << descr; return 1;
    }
    if (nerrs>0) { arg_print_errors(stderr,a_end,(progstr+": "+to_string(__LINE__)+errstr).c_str()); return 1; }


    //Check stdin
    stdi1 = (a_fi->count==0 || strlen(a_fi->filename[0])==0 || strcmp(a_fi->filename[0],"-")==0);
    stdi2 = (a_fi->count<=1 || strlen(a_fi->filename[1])==0 || strcmp(a_fi->filename[1],"-")==0);
    stdi3 = (a_fi->count<=2 || strlen(a_fi->filename[2])==0 || strcmp(a_fi->filename[2],"-")==0);
    stdi4 = (a_fi->count<=3 || strlen(a_fi->filename[3])==0 || strcmp(a_fi->filename[3],"-")==0);
    stdi5 = (a_fi->count<=4 || strlen(a_fi->filename[4])==0 || strcmp(a_fi->filename[4],"-")==0);
    stdi6 = (a_fi->count<=5 || strlen(a_fi->filename[5])==0 || strcmp(a_fi->filename[5],"-")==0);
    stdi7 = (a_fi->count<=6 || strlen(a_fi->filename[6])==0 || strcmp(a_fi->filename[6],"-")==0);
    stdi8 = (a_fi->count<=7 || strlen(a_fi->filename[7])==0 || strcmp(a_fi->filename[7],"-")==0);
    if (stdi1+stdi2+stdi3+stdi4+stdi5+stdi6+stdi7+stdi8>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2+stdi3+stdi4+stdi5+stdi6+stdi7+stdi8>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0 || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open inputs
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 1" << endl; return 1; }
    if (stdi2) { ifs2.copyfmt(cin); ifs2.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs2.open(a_fi->filename[1]); }
    if (!ifs2) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 2" << endl; return 1; }
    if (stdi3) { ifs3.copyfmt(cin); ifs3.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs3.open(a_fi->filename[2]); }
    if (!ifs3) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 3" << endl; return 1; }
    if (stdi4) { ifs4.copyfmt(cin); ifs4.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs4.open(a_fi->filename[3]); }
    if (!ifs4) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 4" << endl; return 1; }
    if (stdi5) { ifs5.copyfmt(cin); ifs5.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs5.open(a_fi->filename[4]); }
    if (!ifs5) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 5" << endl; return 1; }
    if (stdi6) { ifs6.copyfmt(cin); ifs6.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs6.open(a_fi->filename[5]); }
    if (!ifs6) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 6" << endl; return 1; }
    if (stdi7) { ifs7.copyfmt(cin); ifs7.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs7.open(a_fi->filename[6]); }
    if (!ifs7) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 7" << endl; return 1; }
    if (stdi8) { ifs8.copyfmt(cin); ifs8.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs8.open(a_fi->filename[7]); }
    if (!ifs8) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 8" << endl; return 1; }


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if (!read_input_header(ifs3,i3)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 3" << endl; return 1; }
    if (!read_input_header(ifs4,i4)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 4" << endl; return 1; }
    if (!read_input_header(ifs5,i5)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 5" << endl; return 1; }
    if (!read_input_header(ifs6,i6)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 6" << endl; return 1; }
    if (!read_input_header(ifs7,i7)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 7" << endl; return 1; }
    if (!read_input_header(ifs8,i8)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 8" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0 || (i3.T==oktypes).sum()==0 || (i4.T==oktypes).sum()==0 || (i5.T==oktypes).sum()==0 || (i6.T==oktypes).sum()==0 || (i7.T==oktypes).sum()==0 || (i8.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1]) ? "}" : ","); }
        cerr << endl; return 1;
    }


    //Get options

    //Get dim
    if (a_d->count==0) { dim = (i1.C==1u) ? 1 : 0; }
    else if (a_d->ival[0]<0) { cerr << progstr+": " << __LINE__ << errstr << "dim must be nonnegative" << endl; return 1; }
    else { dim = a_d->ival[0]; }
    if (dim>1) { cerr << progstr+": " << __LINE__ << errstr << "dim must be in {0,1}" << endl; return 1; }

    //Get fs
    fs = (a_fs->count>0) ? a_fs->dval[0] : 10000.0;
    if (fs<=0.0) { cerr << progstr+": " << __LINE__ << errstr << "fs (sample rate) must be positive" << endl; return 1; }


    //Checks
    if (i1.T!=i2.T || i1.T!=i3.T || i1.T!=i4.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have same data type" << endl; return 1; }
    if (i1.T!=i5.T || i1.T!=i6.T || i1.T!=i7.T || i1.T!=i8.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have same data type" << endl; return 1; }
    if (i1.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) found to be empty" << endl; return 1; }
    if (!i1.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 1 (Xe) must be a matrix" << endl; return 1; }
    if (i2.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) found to be empty" << endl; return 1; }
    if (!i2.ismat()) { cerr << progstr+": " << __LINE__ << errstr << "input 2 (Xi) must be a matrix" << endl; return 1; }
    if (i3.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (tau) found to be empty" << endl; return 1; }
    if (!i3.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 3 (tau) must be a vector or scalar" << endl; return 1; }
    if (i4.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (alpha) found to be empty" << endl; return 1; }
    if (!i4.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 4 (alpha) must be a vector or scalar" << endl; return 1; }
    if (i5.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (betae) found to be empty" << endl; return 1; }
    if (!i5.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 5 (betae) must be a vector or scalar" << endl; return 1; }
    if (i6.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (betai) found to be empty" << endl; return 1; }
    if (!i6.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 6 (betai) must be a vector or scalar" << endl; return 1; }
    if (i7.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (gammae) found to be empty" << endl; return 1; }
    if (!i7.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 7 (gammae) must be a vector or scalar" << endl; return 1; }
    if (i8.isempty()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (gammai) found to be empty" << endl; return 1; }
    if (!i8.isvec()) { cerr << progstr+": " << __LINE__ << errstr << "input 8 (gammai) must be a vector or scalar" << endl; return 1; }
    if (i1.R!=i2.R || i1.C!=i2.C) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1 (Xe) and 2 (Xi) must have same size" << endl; return 1; }
    if (i3.N()!=1u)
    {
        if ((dim==0 && i3.N()!=i1.R) || (dim==1 && i3.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length tau must equal N (num neurons)" << endl; return 1; }
    }
    if (i4.N()!=1u)
    {
        if ((dim==0 && i4.N()!=i1.R) || (dim==1 && i4.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length alpha must equal N (num neurons)" << endl; return 1; }
    }
    if (i5.N()!=1u)
    {
        if ((dim==0 && i5.N()!=i1.R) || (dim==1 && i5.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length betae must equal N (num neurons)" << endl; return 1; }
    }
    if (i6.N()!=1u)
    {
        if ((dim==0 && i6.N()!=i1.R) || (dim==1 && i6.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length betai must equal N (num neurons)" << endl; return 1; }
    }
    if (i7.N()!=1u)
    {
        if ((dim==0 && i7.N()!=i1.R) || (dim==1 && i7.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length gammae must equal N (num neurons)" << endl; return 1; }
    }
    if (i8.N()!=1u)
    {
        if ((dim==0 && i8.N()!=i1.R) || (dim==1 && i8.N()!=i1.C))
        { cerr << progstr+": " << __LINE__ << errstr << "length gammai must equal N (num neurons)" << endl; return 1; }
    }


    //Set output header info
    o1.F = i1.F; o1.T = i1.T;
    o1.R = i1.R; o1.C = i1.C; o1.S = i1.S; o1.H = i1.H;


    //Open output
    if (wo1)
    {
        if (stdo1) { ofs1.copyfmt(cout); ofs1.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs1.open(a_fo->filename[0]); }
        if (!ofs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 1" << endl; return 1; }
    }


    //Write output header
    if (wo1 && !write_output_header(ofs1,o1)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 1" << endl; return 1; }


    //Other prep
    N = (dim==0) ? int(o1.R) : int(o1.C);
    T = (dim==0) ? int(o1.C) : int(o1.R);
    if (T<2) { cerr << progstr+": " << __LINE__ << errstr << "num time points must be > 1" << endl; return 1; }
    

    //Process
    if (i1.T==1)
    {
        float *Xe, *Xi, *tau, *alpha, *betae, *betai, *gammae, *gammai;
        try { Xe = new float[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (Xe)" << endl; return 1; }
        try { Xi = new float[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xi)" << endl; return 1; }
        try { tau = new float[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (tau)" << endl; return 1; }
        try { alpha = new float[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (alpha)" << endl; return 1; }
        try { betae = new float[i5.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (betae)" << endl; return 1; }
        try { betai = new float[i6.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (betai)" << endl; return 1; }
        try { gammae = new float[i7.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 7 (gammae)" << endl; return 1; }
        try { gammai = new float[i8.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 8 (gammai)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(Xe),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (Xe)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Xi),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xi)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(tau),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (tau)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(alpha),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (alpha)" << endl; return 1; }
        try { ifs5.read(reinterpret_cast<char*>(betae),i5.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (betae)" << endl; return 1; }
        try { ifs6.read(reinterpret_cast<char*>(betai),i6.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (betai)" << endl; return 1; }
        try { ifs7.read(reinterpret_cast<char*>(gammae),i7.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 7 (gammae)" << endl; return 1; }
        try { ifs8.read(reinterpret_cast<char*>(gammai),i8.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 8 (gammai)" << endl; return 1; }
        if (i3.N()==1u) { for (n=1; n<N; n++) { tau[n] = tau[0]; } }
        if (i4.N()==1u) { for (n=1; n<N; n++) { alpha[n] = alpha[0]; } }
        if (i5.N()==1u) { for (n=1; n<N; n++) { betae[n] = betae[0]; } }
        if (i6.N()==1u) { for (n=1; n<N; n++) { betai[n] = betai[0]; } }
        if (i7.N()==1u) { for (n=1; n<N; n++) { gammae[n] = gammae[0]; } }
        if (i8.N()==1u) { for (n=1; n<N; n++) { gammai[n] = gammai[0]; } }
        if (openn::grossberg2_inplace_s(Xe,Xi,tau,alpha,betae,betai,gammae,gammai,N,T,dim,i1.iscolmajor(),float(fs)))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Xe),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] Xe; delete[] Xi; delete[] tau; delete[] alpha; delete[] betae; delete[] betai; delete[] gammae; delete[] gammai;
    }
    else if (i1.T==2)
    {
        double *Xe, *Xi, *tau, *alpha, *betae, *betai, *gammae, *gammai;
        try { Xe = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (Xe)" << endl; return 1; }
        try { Xi = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (Xi)" << endl; return 1; }
        try { tau = new double[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (tau)" << endl; return 1; }
        try { alpha = new double[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (alpha)" << endl; return 1; }
        try { betae = new double[i5.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 5 (betae)" << endl; return 1; }
        try { betai = new double[i6.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 6 (betai)" << endl; return 1; }
        try { gammae = new double[i7.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 7 (gammae)" << endl; return 1; }
        try { gammai = new double[i8.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 8 (gammai)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(Xe),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (Xe)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(Xi),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (Xi)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(tau),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (tau)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(alpha),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (alpha)" << endl; return 1; }
        try { ifs5.read(reinterpret_cast<char*>(betae),i5.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 5 (betae)" << endl; return 1; }
        try { ifs6.read(reinterpret_cast<char*>(betai),i6.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 6 (betai)" << endl; return 1; }
        try { ifs7.read(reinterpret_cast<char*>(gammae),i7.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 7 (gammae)" << endl; return 1; }
        try { ifs8.read(reinterpret_cast<char*>(gammai),i8.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 8 (gammai)" << endl; return 1; }
        if (i3.N()==1u) { for (n=1; n<N; n++) { tau[n] = tau[0]; } }
        if (i4.N()==1u) { for (n=1; n<N; n++) { alpha[n] = alpha[0]; } }
        if (i5.N()==1u) { for (n=1; n<N; n++) { betae[n] = betae[0]; } }
        if (i6.N()==1u) { for (n=1; n<N; n++) { betai[n] = betai[0]; } }
        if (i7.N()==1u) { for (n=1; n<N; n++) { gammae[n] = gammae[0]; } }
        if (i8.N()==1u) { for (n=1; n<N; n++) { gammai[n] = gammai[0]; } }
        if (openn::grossberg2_inplace_d(Xe,Xi,tau,alpha,betae,betai,gammae,gammai,N,T,dim,i1.iscolmajor(),double(fs)))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; } 
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Xe),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] Xe; delete[] Xi; delete[] tau; delete[] alpha; delete[] betae; delete[] betai; delete[] gammae; delete[] gammai;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}

