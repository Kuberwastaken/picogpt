v,u,t,l='lm_head','wpe','wte','input.txt'
s,R,M,L,F,B=isinstance,zip,sum,print,len,range
import os,math as N,random as S
S.seed(42)
if not os.path.exists(l):import urllib.request as P;P.urlretrieve('https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt',l)
O=[x.strip()for x in open(l).read().splitlines()if x.strip()];S.shuffle(O);L(f"num docs: {F(O)}")
T=sorted(set(''.join(O)));U,V=F(T),F(T)+1;L(f"vocab size: {V}")
class E:
	def __init__(A,data,children=(),local_grads=()):A.data=data;A.grad=0;A._children=children;A._local_grads=local_grads
	def __add__(B,other):A=other if s(other,E)else E(other);return E(B.data+A.data,(B,A),(1,1))
	def __mul__(B,other):A=other if s(other,E)else E(other);return E(B.data*A.data,(B,A),(A.data,B.data))
	def __pow__(A,other):return E(A.data**other,(A,),(other*A.data**(other-1),))
	def log(A):return E(N.log(A.data),(A,),(1/A.data,))
	def exp(A):return E(N.exp(A.data),(A,),(N.exp(A.data),))
	def relu(A):return E(max(0,A.data),(A,),(float(A.data>0),))
	def __neg__(A):return A*-1
	__radd__=__add__;__rmul__=__mul__
	def __sub__(A,other):return A+-other
	def __rsub__(A,other):return other+-A
	def __truediv__(A,other):return A*other**-1
	def __rtruediv__(A,other):return other*A**-1
	def backward(B):
		C,D=[],set()
		def E(v):
			if v in D:return
			D.add(v);[E(A)for A in v._children];C.append(v)
		E(B);B.grad=1
		for A in C[::-1]:
			for F,G in R(A._children,A._local_grads):F.grad+=G*A.grad
A,m,J,Y=16,4,1,8;H=A//m
G=lambda nout,nin,std=.02:[[E(S.gauss(0,std))for _ in B(nin)]for _ in B(nout)]
C={t:G(V,A),u:G(Y,A),v:G(V,A)}
for D in B(J):
	p=f"layer{D}."
	C[p+'attn_wq'],C[p+'attn_wk'],C[p+'attn_wv'],C[p+'attn_wo'],C[p+'mlp_fc1'],C[p+'mlp_fc2']=G(A,A),G(A,A),G(A,A),G(A,A,0),G(4*A,A),G(A,4*A,0)
W=[C for A in C.values()for B in A for C in B];L(f"num params: {F(W)}")
def I(x,w):return[M(a*b for a,b in R(r,x))for r in w]
def Z(a):m=max(x.data for x in a);a=[(x-m).exp()for x in a];s=M(a);return[x/s for x in a]
def a(x):s=(M(i*i for i in x)/F(x)+1e-5)**-.5;return[i*s for i in x]
def n(token_id,pos_id,keys,values):
	A=a([i+j for i,j in R(C[t][token_id],C[u][pos_id])])
	for D in B(J):
		G=A;A=a(A);q,k,v0=I(A,C[f"layer{D}.attn_wq"]),I(A,C[f"layer{D}.attn_wk"]),I(A,C[f"layer{D}.attn_wv"]);keys[D].append(k);values[D].append(v0);L0=[]
		for V0 in B(m):
			e=V0*H;w=q[e:e+H];k0=[x[e:e+H]for x in keys[D]];v1=[x[e:e+H]for x in values[D]];p=Z([M(w[i]*k0[j][i]for i in B(H))/H**.5 for j in B(F(k0))]);L0.extend([M(p[j]*v1[j][i]for j in B(F(v1)))for i in B(H)])
		A=[i+j for i,j in R(I(L0,C[f"layer{D}.attn_wo"]),G)];G=A;A=I(a(A),C[f"layer{D}.mlp_fc1"]);A=[x.relu()**2 for x in A];A=[i+j for i,j in R(I(A,C[f"layer{D}.mlp_fc2"]),G)]
	return I(A,C[v])
x,b,c,y=.01,.9,.95,1e-8;d,e=[0.]*F(W),[0.]*F(W);f=500
for P in B(f):
	z=O[P%F(O)];g=[U,*[T.index(A)for A in z],U];o=min(Y,F(g)-1);h,i=[[]for _ in B(J)],[[]for _ in B(J)];p=[]
	for Q in B(o):
		K,A0=g[Q:Q+2];k=Z(n(K,Q,h,i));p.append(-k[A0].log())
	q=M(p)/o;q.backward();A2=x*.5*(1+N.cos(N.pi*P/f))
	for D,X in enumerate(W):
		d[D]=b*d[D]+(1-b)*X.grad;e[D]=c*e[D]+(1-c)*X.grad**2;A3=d[D]/(1-b**(P+1));A4=e[D]/(1-c**(P+1));X.data-=A2*A3/(A4**.5+y);X.grad=0
	L(f"step {P+1:4d} / {f:4d} | loss {q.data:.4f}")
A5=.5;L('\n--- inference ---')
for A6 in B(20):
	h,i=[[]for _ in B(J)],[[]for _ in B(J)];K=U;r=[]
	for Q in B(Y):
		K=S.choices(B(V),weights=[A.data for A in Z([A/A5 for A in n(K,Q,h,i)])])[0]
		if K==U:break
		r.append(T[K])
	L(f"sample {A6+1:2d}: {''.join(r)}")