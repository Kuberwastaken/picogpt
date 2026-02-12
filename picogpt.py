v='lm_head'
u='wpe'
t='wte'
s=isinstance
l='input.txt'
R=zip
M=sum
L=print
F=len
B=range
import os,math as N,random as S
S.seed(42)
if not os.path.exists(l):import urllib.request;w='https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt';urllib.request.urlretrieve(w,l)
O=[A.strip()for A in open(l).read().strip().split('\n')if A.strip()]
S.shuffle(O)
L(f"num docs: {F(O)}")
T=sorted(set(''.join(O)))
U=F(T)
V=F(T)+1
L(f"vocab size: {V}")
class E:
	def __init__(A,data,children=(),local_grads=()):A.data=data;A.grad=0;A._children=children;A._local_grads=local_grads
	def __add__(B,other):A=other;A=A if s(A,E)else E(A);return E(B.data+A.data,(B,A),(1,1))
	def __mul__(B,other):A=other;A=A if s(A,E)else E(A);return E(B.data*A.data,(B,A),(A.data,B.data))
	def __pow__(A,other):B=other;return E(A.data**B,(A,),(B*A.data**(B-1),))
	def log(A):return E(N.log(A.data),(A,),(1/A.data,))
	def exp(A):return E(N.exp(A.data),(A,),(N.exp(A.data),))
	def relu(A):return E(max(0,A.data),(A,),(float(A.data>0),))
	def __neg__(A):return A*-1
	def __radd__(A,other):return A+other
	def __sub__(A,other):return A+-other
	def __rsub__(A,other):return other+-A
	def __rmul__(A,other):return A*other
	def __truediv__(A,other):return A*other**-1
	def __rtruediv__(A,other):return other*A**-1
	def backward(B):
		C=[];D=set()
		def E(v):
			if v not in D:
				D.add(v)
				for A in v._children:E(A)
				C.append(v)
		E(B);B.grad=1
		for A in reversed(C):
			for(F,G)in R(A._children,A._local_grads):F.grad+=G*A.grad
A=16
m=4
J=1
Y=8
H=A//m
G=lambda nout,nin,std=.02:[[E(S.gauss(0,std))for A in B(nin)]for A in B(nout)]
C={t:G(V,A),u:G(Y,A),v:G(V,A)}
for D in B(J):C[f"layer{D}.attn_wq"]=G(A,A);C[f"layer{D}.attn_wk"]=G(A,A);C[f"layer{D}.attn_wv"]=G(A,A);C[f"layer{D}.attn_wo"]=G(A,A,std=0);C[f"layer{D}.mlp_fc1"]=G(4*A,A);C[f"layer{D}.mlp_fc2"]=G(A,4*A,std=0)
W=[C for A in C.values()for B in A for C in B]
L(f"num params: {F(W)}")
def I(x,w):return[M(A*B for(A,B)in R(A,x))for A in w]
def Z(logits):A=logits;C=max(A.data for A in A);B=[(A-C).exp()for A in A];D=M(B);return[A/D for A in B]
def a(x):A=M(A*A for A in x)/F(x);B=(A+1e-05)**-.5;return[A*B for A in x]
def n(token_id,pos_id,keys,values):
	K=values;P=C[t][token_id];Q=C[u][pos_id];A=[A+B for(A,B)in R(P,Q)];A=a(A)
	for D in B(J):
		G=A;A=a(A);S=I(A,C[f"layer{D}.attn_wq"]);T=I(A,C[f"layer{D}.attn_wk"]);U=I(A,C[f"layer{D}.attn_wv"]);keys[D].append(T);K[D].append(U);L=[]
		for V in B(m):E=V*H;W=S[E:E+H];N=[A[E:E+H]for A in keys[D]];O=[A[E:E+H]for A in K[D]];X=[M(W[A]*N[C][A]for A in B(H))/H**.5 for C in B(F(N))];Y=Z(X);b=[M(Y[A]*O[A][C]for A in B(F(O)))for C in B(H)];L.extend(b)
		A=I(L,C[f"layer{D}.attn_wo"]);A=[A+B for(A,B)in R(A,G)];G=A;A=a(A);A=I(A,C[f"layer{D}.mlp_fc1"]);A=[A.relu()**2 for A in A];A=I(A,C[f"layer{D}.mlp_fc2"]);A=[A+B for(A,B)in R(A,G)]
	c=I(A,C[v]);return c
x,b,c,y=.01,.9,.95,1e-08
d=[.0]*F(W)
e=[.0]*F(W)
f=500
for P in B(f):
	z=O[P%F(O)];g=[U]+[T.index(A)for A in z]+[U];o=min(Y,F(g)-1);h,i=[[]for A in B(J)],[[]for A in B(J)];p=[]
	for Q in B(o):K,A0=g[Q],g[Q+1];j=n(K,Q,h,i);k=Z(j);A1=-k[A0].log();p.append(A1)
	q=1/o*M(p);q.backward();A2=x*.5*(1+N.cos(N.pi*P/f))
	for(D,X)in enumerate(W):d[D]=b*d[D]+(1-b)*X.grad;e[D]=c*e[D]+(1-c)*X.grad**2;A3=d[D]/(1-b**(P+1));A4=e[D]/(1-c**(P+1));X.data-=A2*A3/(A4**.5+y);X.grad=0
	L(f"step {P+1:4d} / {f:4d} | loss {q.data:.4f}")
A5=.5
L('\n--- inference ---')
for A6 in B(20):
	h,i=[[]for A in B(J)],[[]for A in B(J)];K=U;r=[]
	for Q in B(Y):
		j=n(K,Q,h,i);k=Z([A/A5 for A in j]);K=S.choices(B(V),weights=[A.data for A in k])[0]
		if K==U:break
		r.append(T[K])
	L(f"sample {A6+1:2d}: {"".join(r)}")