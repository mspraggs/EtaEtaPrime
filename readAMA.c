#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
/* Andreas Juettner, 03/2013

 This code takes CPS ASCII AMA correlators and exact correlators
 as input and constructs 
 - AMA
 - residual
 - sloppy
 - exact 
 correlator averages

 Currently this is implemented for meson 2point 
 and 3point functions
 
 TODO: - other contractions
       - tidy up
       - 2pt and 3pt function could potentially be merged
         into one versatile function
       - currently code assumes that there are TT (= T/a)
         sloppy solves and this is hard coded. This might
         turn out to be too restricitve in the future
*/


/******************************************************************************/
void average(double *out, const double *in,int N,int n){
 /* takes a vector of length N, cut's it into n pieces,
    averages the result; output is vector of length N/n */
 int i,j; 
 double norm;
 if (N%n){
  printf("readAMA::average: N=%d not divisible by n=%d",N,n);
  exit(0);
  }
 for (j=0;j<N/n;j++){
  out[j]=0.;
 }
 norm = 1./((double)n);
 for (i=0;i<n;i++){
  for (j=0;j<N/n;j++){
   out[j]+=norm*in[N/n*i+j];
  }
 }
}
/******************************************************************************/
int find(int Ntt, int *tsrc_EA,int n){
 /* find first element of value n in array tsrc_exact
    and return position of this entry */
 int i;
 for (i=0;i<Ntt;i++){
  if (tsrc_EA[i]==n){
   return i;}
 }
 return -1;
}
/******************************************************************************/
int find2(int TT,int Ntt, int Ndeltat, 
	int *tsrc_EA, int *tsnk_EA,
	int  isrc,    int  isnk){
 /* find first element with tsrc_exact == isrc and tsnk_exact==isnk
    and return position of this entry */
 int i,j,k;
 for (i=0;i<Ndeltat;i++){
  for (j=0;j<Ntt;j++){
   k = i*Ntt + j ;
   if (tsrc_EA[k]==isrc && tsnk_EA[k]==isnk){
    return k;}
  }
 }
 return -1;
}
/******************************************************************************/

void read2pt(char *filenameEA,char *filenameA,int TT,int Ntt){
 /* this is the code for 2pt functions */
 int i,n,flag,offset;
 char fnA[512],fnEA[512];
 FILE *fEA=NULL,*fA=NULL;
 int tsrc,t;
 int *tsrc_EA=NULL,nval_EA=0,nsrc_EA=0;
 int *tsrc_A=NULL ,nval_A=0,nsrc_A =0;
 double *re_EA=NULL,*im_EA=NULL;
 double *re_A=NULL,*im_A=NULL;
 double *re_S=NULL,*im_S=NULL;
 double *re_R=NULL,*im_R=NULL;
 double *re_E=NULL,*im_E=NULL;
 double *re_AMA=NULL,*im_AMA=NULL;
 long double re,im;
 double norm;
 norm=1./((double)Ntt);
 /* allocate space for 
    - Sloppy
    - Residual
    - Exact
    - AMA 
    correlators for real and imag part */
 re_S  =malloc(TT*sizeof(double));
 re_R  =malloc(TT*sizeof(double));
 re_E  =malloc(TT*sizeof(double));
 re_AMA=malloc(TT*sizeof(double));
 im_S  =malloc(TT*sizeof(double));
 im_R  =malloc(TT*sizeof(double));
 im_E  =malloc(TT*sizeof(double));
 im_AMA=malloc(TT*sizeof(double));

 sprintf(fnEA,"%s",filenameEA);
 sprintf(fnA,"%s",filenameA);

 fEA = fopen(fnEA,"r") ; /* read in exact data */
 if(fEA == (FILE *)NULL){
  printf("Can't open config file %s",fnEA);
  exit(3);
 }

 /* read in the exact correlators */
 while( (!feof(fEA))){
  fscanf(fEA, "%d%d%Le%Le",&tsrc,&t,&re,&im);
  if (nval_EA%TT==0 && t==0){
   tsrc_EA = (int *) realloc (tsrc_EA,(nsrc_EA+1) * sizeof(int *));
   tsrc_EA[nsrc_EA] = tsrc;
   nsrc_EA++;
  }
  re_EA = (double *) realloc (re_EA,(nval_EA+1) * sizeof(double *));
  im_EA = (double *) realloc (im_EA,(nval_EA+1) * sizeof(double *));
  re_EA[nval_EA] = (double)re;
  im_EA[nval_EA] = (double)im;
  nval_EA++;
 }

 fclose(fEA);

 if (Ntt!=(nval_EA)/TT){
  printf("readAMA:main: sth. wrong here - Ntt!=(nval_EA-1)/TT\n");
  printf("              Ntt=%d nval_EA=%d TT=%d\n",Ntt,nval_EA,TT);
  exit(0);
  }
 /* average the exact correlators over Ntt source positions */
 average(re_E,re_EA,TT*Ntt,Ntt);
 average(im_E,im_EA,TT*Ntt,Ntt);
  
 fA = fopen(fnA,"r") ;
 if(fA == (FILE *)NULL){
  printf("Can't open config file %s",fnA);
  exit(3);
 }
 
 /* read in the sloppy correlators */
 while( (!feof(fA))){
  fscanf(fA, "%d%d%Le%Le",&tsrc,&t,&re,&im);
  if (nval_A%TT==0 && t==0){
   tsrc_A = (int *) realloc (tsrc_A,(nsrc_A+1) * sizeof(int *));
   tsrc_A[nsrc_A] = tsrc;
   n=find(Ntt,tsrc_EA,tsrc); 
   if (n==-1) flag=0; /* set a flag if there exists an exact solve */
   else flag=1;       /* for the current source position */
   nsrc_A++; 
  }
  re_A = (double *) realloc (re_A,(nval_A+1) * sizeof(double *));
  im_A = (double *) realloc (im_A,(nval_A+1) * sizeof(double *));
  re_A[nval_A] = (double)re;
  im_A[nval_A] = (double)im;
  nval_A++;
  if (flag){ /* subtract sloppy result from exact result if 
	        the latter exists for the current source position */
   offset = TT*n;
   re_EA[offset+t]-=re;
   im_EA[offset+t]-=im;
  }
 }
 fclose(fA);
 
 average(re_S,re_A,TT*TT,TT);    /* average sloppy solves over TT source  */
 average(im_S,im_A,TT*TT,TT);    /* positions (TT is hard coded here since
				    this is what we are currently doing */
 average(re_R,re_EA,TT*Ntt,Ntt); /* average residual over Ntt source */
 average(im_R,im_EA,TT*Ntt,Ntt); /* positions */

 for (i=0;i<TT;i++){
  re_AMA[i]=re_R[i]+re_S[i];  /* construct AMA correlators */
  im_AMA[i]=im_R[i]+im_S[i];
  printf("%2d %2d %+e %+e %+e %+e %+e %+e %+e %+e\n",
	  i,0,re_E[i],im_E[i],
          re_S[i],im_S[i],
          re_R[i],im_R[i],
          re_AMA[i],im_AMA[i]);
  }
 }
/******************************************************************************/
void read3pt(char *filenameEA,char *filenameA,int TT,int NdeltaT,int Ntt){
 /* this is the code for 3pt functions */
 int i,j,k,n,m,flag=0,dT,idT;
 char fnA[512],fnEA[512];
 FILE *fEA=NULL,*fA=NULL;
 int tsrc_m_tsnk,tsnk,t;
 int *tsrc_EA=NULL,nval_EA=0,nsrc_EA=0;
 int *tsnk_EA=NULL,nsnk_EA=0;
 int nval_A=0,nsrc_A =0;
 double **re_EA=NULL ,**im_EA=NULL;
 double **re_A=NULL  ,**im_A=NULL;
 double **re_S=NULL  ,**im_S=NULL;
 double **re_R=NULL  ,**im_R=NULL;
 double **re_E=NULL  ,**im_E=NULL;
 double **re_AMA=NULL,**im_AMA=NULL;
 long double re0,im0;
 long double re1,im1;
 long double re2,im2;
 long double re3,im3;
 long double re4,im4;
 int *dTcount;
 double norm;
 norm=1./((double)Ntt);

 dTcount= malloc(TT*sizeof(int));
 for (i=0;i<TT;i++) dTcount[i]=0.;
 /* allocate memory space for x,y,z,t,1 vector current for 
    - Sloppy
    - Residual
    - Exact
    - AMA 
    correlators for real and imag part */
 re_A  =malloc(5*sizeof(double));
 re_EA =malloc(5*sizeof(double));
 re_S  =malloc(5*sizeof(double));
 re_R  =malloc(5*sizeof(double));
 re_E  =malloc(5*sizeof(double));
 re_AMA=malloc(5*sizeof(double));
 im_A  =malloc(5*sizeof(double));
 im_EA =malloc(5*sizeof(double));
 im_S  =malloc(5*sizeof(double));
 im_R  =malloc(5*sizeof(double));
 im_E  =malloc(5*sizeof(double));
 im_AMA=malloc(5*sizeof(double));
 for (i=0;i<5;i++){
  re_A[i]  =malloc(TT*TT*sizeof(double));
  re_EA[i] =malloc(TT*TT*sizeof(double));
  re_S[i]  =malloc(TT*TT*sizeof(double));
  re_R[i]  =malloc(TT*TT*sizeof(double));
  re_E[i]  =malloc(TT*TT*sizeof(double));
  re_AMA[i]=malloc(TT*TT*sizeof(double));
  im_A[i]  =malloc(TT*TT*sizeof(double));
  im_EA[i] =malloc(TT*TT*sizeof(double));
  im_S[i]  =malloc(TT*TT*sizeof(double));
  im_R[i]  =malloc(TT*TT*sizeof(double));
  im_E[i]  =malloc(TT*TT*sizeof(double));
  im_AMA[i]=malloc(TT*TT*sizeof(double));
 }

 sprintf(fnA,"%s",filenameA);
 sprintf(fnEA,"%s",filenameEA);

 fEA = fopen(fnEA,"r") ; /* read in exact data */
 if(fEA == (FILE *)NULL){
  printf("Can't open config file %s",fnEA);
  exit(3);
 }

 /* read in the exact propagators */
 while( (!feof(fEA))){
  fscanf(fEA, "%d%d%d%Le%Le%Le%Le%Le%Le%Le%Le%Le%Le",
	&tsrc_m_tsnk,&tsnk,&t,
	&re0,&im0,
	&re1,&im1,
	&re2,&im2,
	&re3,&im3,
	&re4,&im4);
  if ((nval_EA)%TT==0 && t==0){
   /* Hantao's output contains:
      - 1st column: source-sink separation
      - 2nd column: sink location
      - 3rd column: operator insation time slice */
   tsrc_EA = (int *) realloc (tsrc_EA,(nsrc_EA+1) * sizeof(int *));
   tsrc_EA[nsrc_EA] = (tsrc_m_tsnk+tsnk);
   tsnk_EA = (int *) realloc (tsnk_EA,(nsnk_EA+1) * sizeof(int *));
   tsnk_EA[nsnk_EA] = (tsnk);
   /* count the multiplicity of a given source-sink separation: */
   dTcount[tsrc_EA[nsrc_EA]-tsnk_EA[nsnk_EA]]+=1; 
						  
   nsrc_EA++;
   nsnk_EA++;
  }
  for (i=0;i<5;i++){
   re_EA[i] = (double *) realloc (re_EA[i],(nval_EA+1) * sizeof(double *));
   im_EA[i] = (double *) realloc (im_EA[i],(nval_EA+1) * sizeof(double *));
  }
  /* assign real and imaginary parts of exact correlator:
     x,y,z,t component of the vector current and also the scalar current */
  re_EA[0][nval_EA] = (double)re0;
  im_EA[0][nval_EA] = (double)im0;
  re_EA[1][nval_EA] = (double)re1;
  im_EA[1][nval_EA] = (double)im1;
  re_EA[2][nval_EA] = (double)re2;
  im_EA[2][nval_EA] = (double)im2;
  re_EA[3][nval_EA] = (double)re3;
  im_EA[3][nval_EA] = (double)im3;
  re_EA[4][nval_EA] = (double)re4;
  im_EA[4][nval_EA] = (double)im4;
  nval_EA++;
 }
 fclose(fEA);
 if (Ntt!=(nval_EA-1)/NdeltaT/TT){
  printf("readAMA:main: sth. wrong here - Ntt!=(nval_EA-1)/TT/NdeltaT\n");
  printf("readAMA:main: NdeltaT=%d Ntt=%d, nval_EA=%d, TT=%d\n",
			NdeltaT,Ntt,nval_EA-1,TT);
  exit(0);
  }
 for (j=0;j<5;j++){
 i=0;
 m=0;
 k=0;
 for (i=0;i<(nval_EA-1)/TT;i++){
  dT=dTcount[tsrc_EA[i]-tsnk_EA[i]];
  if (dT){
  average(re_E[j]+k*TT,re_EA[j]+m,TT*dT,dT); /* av. exact over dT
  average(im_E[j]+k*TT,im_EA[j]+m,TT*dT,dT);    src-snk separations */
  m+=TT*dT; 
  i+=dT-1; /* set counter but subtract one which the loop adds again */
  k++;
  }
 }
 }

 /* open file with sloppy data */
 sprintf(fnA,"%s",filenameA);
 fA = fopen(fnA,"r");
 if(fA == (FILE *)NULL){
  printf("Can't open config file %s",fnA);
  exit(3);
 }

 /* read in the sloppy propagators */
 while( (!feof(fA))){
  fscanf(fA, "%d%d%d%Le%Le%Le%Le%Le%Le%Le%Le%Le%Le",
	&tsrc_m_tsnk,&tsnk,&t,&re0,&im0,
 			      &re1,&im1,
			      &re2,&im2,
			      &re3,&im3,
			      &re4,&im4);

  if (nval_A%TT==0 && t==0){ /* this triggers for each new source position */
   n=find2(TT,Ntt,NdeltaT,tsrc_EA,tsnk_EA,tsrc_m_tsnk+tsnk,tsnk); 
					 /* check whether current 
					    source position corresponds
					    to exact solve, if so
					    n is the position */
   if (n==-1) flag=0;
   else flag=1;
   nsrc_A++;
  }
  for (i=0;i<5;i++){
   re_A[i] = (double *) realloc (re_A[i],(nval_A+1) * sizeof(double *));
   im_A[i] = (double *) realloc (im_A[i],(nval_A+1) * sizeof(double *));
  }
  /* assign real and imaginary parts of sloppy correlator:
     x,y,z,t component of the vector current and also the scalar current */
  re_A[0][nval_A] = (double)re0;
  im_A[0][nval_A] = (double)im0;
  re_A[1][nval_A] = (double)re1;
  im_A[1][nval_A] = (double)im1;
  re_A[2][nval_A] = (double)re2;
  im_A[2][nval_A] = (double)im2;
  re_A[3][nval_A] = (double)re3;
  im_A[3][nval_A] = (double)im3;
  re_A[4][nval_A] = (double)re4;
  im_A[4][nval_A] = (double)im4;
  if (flag){
   re_EA[0][n*TT+t]-=re0; /* subtract sloppy solve on current time slice */
   im_EA[0][n*TT+t]-=im0; /* of result from same source position */
   re_EA[1][n*TT+t]-=re1;
   im_EA[1][n*TT+t]-=im1;
   re_EA[2][n*TT+t]-=re2;
   im_EA[2][n*TT+t]-=im2;
   re_EA[3][n*TT+t]-=re3;
   im_EA[3][n*TT+t]-=im3;
   re_EA[4][n*TT+t]-=re4;
   im_EA[4][n*TT+t]-=im4;
  }
  nval_A++;
 }

 for (j=0;j<5;j++){
  i=0;
  m=0;
  k=0;
  for (i=0;i<(nval_EA-1)/TT;i++){ 
   dT = tsrc_EA[i]-tsnk_EA[i];
   idT=dTcount[dT]; /* multiplicity of source-sink sep. */
   if (idT){
    /* Average the residual (which may exists on a limited # of time slices) */
    average(re_R[j]+k*TT,re_EA[j]+m       ,TT*idT,idT);  
    average(im_R[j]+k*TT,im_EA[j]+m       ,TT*idT,idT);  
    /* Average sloppy solves (which exists for solves on all timeslices) */
    average(re_S[j]+k*TT,re_A[j] +dT*TT*TT,TT*TT ,TT); 
    average(im_S[j]+k*TT,im_A[j] +dT*TT*TT,TT*TT ,TT);
    m+=TT*idT; 
    i+=idT-1;
    for (t=k*TT;t<(k+1)*TT;t++){
     re_AMA[j][t]=re_R[j][t]+re_S[j][t];
     im_AMA[j][t]=im_R[j][t]+im_S[j][t];
     printf("%2d %2d %+e %+e %+e %+e %+e %+e %+e %+e\n",
     t,dT,
          re_E[j][t],im_E[j][t],
          re_S[j][t],im_S[j][t],
          re_R[j][t],im_R[j][t],
          re_AMA[j][t],im_AMA[j][t]);
   }
   k++;
  }
 }
 }
}
void printinfo(){
  printf(" \n");
  printf(" juettner@soton.ac.uk, 03/2013\n");
  printf(" \n");
  printf("readAMA <datafile exact> <datafile sloppy> <type> <TT> <Nsrc> <NdeltaT>\n");
  printf(" <type> = 2,3: 2pt, 3pt function\n");
  printf(" <TT>        : time extent\n");
  printf(" <Nsrc>      : number of sources\n");
  printf(" <NdeltaT>   : number of src-snk separations (only applies to\n");
  printf("               3pt function\n");
  printf("\n");
  printf("this code takes Hantao's ASCII data for exact and sloppy 2pt-\n");
  printf("and 3pt-functions (Kl3) and carries out the AMA-construction.\n");
  printf("The output will contain 9 columns:\n");
  printf("  0   : line number\n");
  printf("  1   : src-snk separation (applies to 3pt function only)\n");
  printf("  2,3 : re,im exact solve, averaged over Ntt source positions\n");
  printf("  4,5 : re,im sloppy solve,    -- \" -- \n");
  printf("  6,7 : re,im residual    ,    -- \" -- \n");
  printf("  8,9 : re,im AMA         ,    -- \" -- \n");
  printf(" \n");
  printf("If type is 3, i.e. 3pt function, the output contains the same\n");
  printf("column format and there are 5 consecutive blocks of correlators.\n");
  printf("One block contains the correlators for different source-sink\n");
  printf("separations and the consecutive blocks \n");
  printf("correspond to V_x,V_y,V_z,V_t,S as the current insertion\n");
  printf("\n");
  printf("02.04.2013, AJ: This code for meson 2pt and tree-level 3pt functions\n");
  printf("                has been cross-checked with Hantao's\n");
  printf("                routine for doing the AMA construction \n");
}

/**********************************************************************/ 
int main(int argc,char*argv[]){
 int Ntt,NdeltaT,TT,type;

 
 if (argc == 0){
  printinfo();
  printf("chello %d %s %d\n",argc, argv[1],strcmp(argv[1],"\?"));
 }
 if (argc > 0 && (argc==7 || argc ==6)){
  if (atoi(argv[3])==2){
   type    = atoi(argv[3]);
   TT      = atoi(argv[4]);
   Ntt     = atoi(argv[5]);
   read2pt(argv[1],argv[2],TT,Ntt);
  }
  else if (argc == 7 && atoi(argv[3])==3){
   type    = atoi(argv[3]);
   TT      = atoi(argv[4]);
   Ntt     = atoi(argv[5]);
   NdeltaT = atoi(argv[6]); 
   read3pt(argv[1],argv[2],TT,NdeltaT,Ntt);
  }
  else {
   printinfo();
  }
 }
 else
   printinfo();
 return 0;
 }

