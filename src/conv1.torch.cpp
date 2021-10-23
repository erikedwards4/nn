//@author Erik Edwards
//@date 2018-present
//@license BSD 3-clause


#include <iostream>
#include <fstream>
#include <unistd.h>
#include <string>
#include <cstring>
#include <valarray>
#include <unordered_map>
#include <argtable2.h>
#include "../util/cmli.hpp"
#include "conv1.torch.c"

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
    const valarray<size_t> oktypes = {1u,2u};
    const size_t I = 3u, O = 1u;
    ifstream ifs1, ifs2, ifs3; ofstream ofs1;
    int8_t stdi1, stdi2, stdi3, stdo1, wo1;
    ioinfo i1, i2, i3, o1;
    size_t Nb, Ni, No, Li, Lo, Lk, str;
    int pad, Ti, ceil_mode, pm;
    string pad_mode;


    //Description
    string descr;
    descr += "1D convolution using PyTorch shape conventions.\n";
    descr += "All inputs/outputs are row-major.\n";
    descr += "\n";
    descr += "X has size Nb x Ni x Li,\n";
    descr += "where:\n";
    descr += "Nb is the batch size.\n";
    descr += "Ni is the number of input neurons (C_in).\n";
    descr += "Li is the input length (usually the number of time points).\n";
    descr += "\n";
    descr += "K is the tensor of convolving kernels.\n";
    descr += "K has size No x Ni x Lk,\n";
    descr += "where:\n";
    descr += "Lk is the kernel length (kernel_size, or time width).\n";
    descr += "\n";
    descr += "B is the bias vector of length No,\n";
    descr += "where:\n";
    descr += "No is the number of output neurons (C_out).\n";
    descr += "\n";
    descr += "Y has size Nb x No x Lo,\n";
    descr += "where:\n";
    descr += "Lo =  ceil[1 + (Li + 2*pad - Lk)/stride], if ceil_mode=true.\n";
    descr += "Lo = floor[1 + (Li + 2*pad - Lk)/stride], if ceil_mode=false.\n";
    descr += "\n";
    descr += "If Nb==1, then X can be entered as a matrix with size Ni x Li,\n";
    descr += "and then Y will be a matrix with size No x Lo.\n";
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
    descr += "$ conv1.torch X K B -o Y \n";
    descr += "$ conv1.torch -s2 -p10 -m'reflect' X K B > Y \n";
    descr += "$ cat X | conv1.torch -c -s5 -p10 -m'c' - K B > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X,K,B)");
    struct arg_int  *a_str = arg_intn("s","stride","<uint>",0,1,"stride in samps [default=1]");
    struct arg_int  *a_pad = arg_intn("p","padding","<int>",0,1,"padding [default=0]");
    struct arg_str   *a_pm = arg_strn("m","pad_mode","<str>",0,1,"padding mode [default='zeros']");
    struct arg_lit   *a_cm = arg_litn("c","ceil_mode",0,1,"include to use ceil_mode [default=false]");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_str, a_pad, a_pm, a_cm, a_fo, a_help, a_end};
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
    stdi1 = (a_fi->count==0 || strlen(a_fi->filename[0])==0u || strcmp(a_fi->filename[0],"-")==0);
    stdi2 = (a_fi->count<=1 || strlen(a_fi->filename[1])==0u || strcmp(a_fi->filename[1],"-")==0);
    stdi3 = (a_fi->count<=2 || strlen(a_fi->filename[2])==0u || strcmp(a_fi->filename[2],"-")==0);
    if (stdi1+stdi2+stdi3>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2+stdi3>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


    //Check stdout
    if (a_fo->count>0) { stdo1 = (strlen(a_fo->filename[0])==0u || strcmp(a_fo->filename[0],"-")==0); }
    else { stdo1 = (!isatty(fileno(stdout))); }
    wo1 = (stdo1 || a_fo->count>0);


    //Open inputs
    if (stdi1) { ifs1.copyfmt(cin); ifs1.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs1.open(a_fi->filename[0]); }
    if (!ifs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 1" << endl; return 1; }
    if (stdi2) { ifs2.copyfmt(cin); ifs2.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs2.open(a_fi->filename[1]); }
    if (!ifs2) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 2" << endl; return 1; }
    if (stdi3) { ifs3.copyfmt(cin); ifs3.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs3.open(a_fi->filename[2]); }
    if (!ifs3) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 3" << endl; return 1; }


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if (!read_input_header(ifs3,i3)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 3" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0 || (i3.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1u]) ? "}" : ","); }
        cerr << endl; return 1;
    }


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
    if (i1.ismat()) { Nb = 1u; Ni = i1.R; Li = i1.C;}
    else { Nb = i1.R; Ni = i1.C; Li = i1.S; }
    No = i2.R; Lk = i2.S;
    Ti = (int)Li + 2*pad;
    if (i1.T!=i2.T || i1.T!=i3.T) { cerr << progstr+": " << __LINE__ << errstr << "inputs must have the same data type" << endl; return 1; }
    if (i1.iscolmajor() || i2.iscolmajor()) { cerr << progstr+": " << __LINE__ << errstr << "inputs 1 and 2 must have row-major layout" << endl; return 1; }
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
    if (i1.ismat()) { o1.R = No; o1.C = Lo; o1.S = 1u; }
    else { o1.R = Nb; o1.C = No; o1.S = Lo; }
    o1.H = 1u;


    //Open output
    if (wo1)
    {
        if (stdo1) { ofs1.copyfmt(cout); ofs1.basic_ios<char>::rdbuf(cout.rdbuf()); } else { ofs1.open(a_fo->filename[0]); }
        if (!ofs1) { cerr << progstr+": " << __LINE__ << errstr << "problem opening output file 1" << endl; return 1; }
    }


    //Write output header
    if (wo1 && !write_output_header(ofs1,o1)) { cerr << progstr+": " << __LINE__ << errstr << "problem writing header for output file 1" << endl; return 1; }


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
        if (codee::conv1_torch_s(Y,X,K,B,Nb,Ni,No,Li,Lk,pad,str,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] K; delete[] B; delete[] Y;
    }
    else if (i1.T==2u)
    {
        double *X, *K, *B, *Y;
        try { X = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X)" << endl; return 1; }
        try { K = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (K)" << endl; return 1; }
        try { B = new double[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (B)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(K),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (K)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(B),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (B)" << endl; return 1; }
        if (codee::conv1_torch_d(Y,X,K,B,Nb,Ni,No,Li,Lk,pad,str,ceil_mode,pm))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X; delete[] K; delete[] B; delete[] Y;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    
    //Close fstreams
    ifs1.close(); ifs2.close();
 ifs3.close();

    ofs1.close();

    //Exit
    return ret;
}
