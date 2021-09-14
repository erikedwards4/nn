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
#include "bilinear.cblas.c"

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
    const valarray<size_t> oktypes = {1u,2u,101u,102u};
    const size_t I = 4u, O = 1u;
    ifstream ifs1, ifs2, ifs3, ifs4; ofstream ofs1;
    int8_t stdi1, stdi2, stdi3, stdi4, stdo1, wo1;
    ioinfo i1, i2, i3, i4, o1;
    size_t Ni1, Ni2, No, L;


    //Description
    string descr;
    descr += "IN method.\n";
    descr += "Bilinear transformation of 2 layers of inputs using CBLAS.\n";
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
    descr += "$ bilinear.cblas X1 X2 W B -o Y \n";
    descr += "$ bilinear.cblas X1 X2 W B > Y \n";
    descr += "$ cat X | bilinear.cblas - W B > Y \n";


    //Argtable
    int nerrs;
    struct arg_file  *a_fi = arg_filen(nullptr,nullptr,"<file>",I-1,I,"input files (X1,X2,W,B)");
    struct arg_file  *a_fo = arg_filen("o","ofile","<file>",0,O,"output file (Y)");
    struct arg_lit *a_help = arg_litn("h","help",0,1,"display this help and exit");
    struct arg_end  *a_end = arg_end(5);
    void *argtable[] = {a_fi, a_fo, a_help, a_end};
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
    stdi4 = (a_fi->count<=3 || strlen(a_fi->filename[3])==0u || strcmp(a_fi->filename[3],"-")==0);
    if (stdi1+stdi2+stdi3+stdi4>1) { cerr << progstr+": " << __LINE__ << errstr << "can only use stdin for one input" << endl; return 1; }
    if (stdi1+stdi2+stdi3+stdi4>0 && isatty(fileno(stdin))) { cerr << progstr+": " << __LINE__ << errstr << "no stdin detected" << endl; return 1; }


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
    if (stdi4) { ifs4.copyfmt(cin); ifs4.basic_ios<char>::rdbuf(cin.rdbuf()); } else { ifs4.open(a_fi->filename[3]); }
    if (!ifs4) { cerr << progstr+": " << __LINE__ << errstr << "problem opening input file 4" << endl; return 1; }


    //Read input headers
    if (!read_input_header(ifs1,i1)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 1" << endl; return 1; }
    if (!read_input_header(ifs2,i2)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 2" << endl; return 1; }
    if (!read_input_header(ifs3,i3)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 3" << endl; return 1; }
    if (!read_input_header(ifs4,i4)) { cerr << progstr+": " << __LINE__ << errstr << "problem reading header for input file 4" << endl; return 1; }
    if ((i1.T==oktypes).sum()==0 || (i2.T==oktypes).sum()==0 || (i3.T==oktypes).sum()==0 || (i4.T==oktypes).sum()==0)
    {
        cerr << progstr+": " << __LINE__ << errstr << "input data type must be in " << "{";
        for (auto o : oktypes) { cerr << int(o) << ((o==oktypes[oktypes.size()-1u]) ? "}" : ","); }
        cerr << endl; return 1;
    }


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
    L = i1.isvec() ? 1u : i1.iscolmajor() ? i1.C : i1.R;
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
        if (codee::bilinear_cblas_s(Y,X1,X2,W,B,Ni1,Ni2,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X1; delete[] X2; delete[] W; delete[] B; delete[] Y;
    }
    else if (i1.T==2u)
    {
        double *X1, *X2, *W, *B, *Y;
        try { X1 = new double[i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
        try { X2 = new double[i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
        try { W = new double[i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (W)" << endl; return 1; }
        try { B = new double[i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (B)" << endl; return 1; }
        try { Y = new double[o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(W),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (W)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(B),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (B)" << endl; return 1; }
        if (codee::bilinear_cblas_d(Y,X1,X2,W,B,Ni1,Ni2,No,L))
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
        if (codee::bilinear_cblas_c(Y,X1,X2,W,B,Ni1,Ni2,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X1; delete[] X2; delete[] W; delete[] B; delete[] Y;
    }
    else if (i1.T==102u)
    {
        double *X1, *X2, *W, *B, *Y;
        try { X1 = new double[2u*i1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 1 (X1)" << endl; return 1; }
        try { X2 = new double[2u*i2.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 2 (X2)" << endl; return 1; }
        try { W = new double[2u*i3.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 3 (W)" << endl; return 1; }
        try { B = new double[2u*i4.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for input file 4 (B)" << endl; return 1; }
        try { Y = new double[2u*o1.N()]; }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem allocating for output file (Y)" << endl; return 1; }
        try { ifs1.read(reinterpret_cast<char*>(X1),i1.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 1 (X1)" << endl; return 1; }
        try { ifs2.read(reinterpret_cast<char*>(X2),i2.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 2 (X2)" << endl; return 1; }
        try { ifs3.read(reinterpret_cast<char*>(W),i3.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 3 (W)" << endl; return 1; }
        try { ifs4.read(reinterpret_cast<char*>(B),i4.nbytes()); }
        catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem reading input file 4 (B)" << endl; return 1; }
        if (codee::bilinear_cblas_z(Y,X1,X2,W,B,Ni1,Ni2,No,L))
        { cerr << progstr+": " << __LINE__ << errstr << "problem during function call" << endl; return 1; }
        if (wo1)
        {
            try { ofs1.write(reinterpret_cast<char*>(Y),o1.nbytes()); }
            catch (...) { cerr << progstr+": " << __LINE__ << errstr << "problem writing output file (Y)" << endl; return 1; }
        }
        delete[] X1; delete[] X2; delete[] W; delete[] B; delete[] Y;
    }
    else
    {
        cerr << progstr+": " << __LINE__ << errstr << "data type not supported" << endl; return 1;
    }
    

    //Exit
    return ret;
}
