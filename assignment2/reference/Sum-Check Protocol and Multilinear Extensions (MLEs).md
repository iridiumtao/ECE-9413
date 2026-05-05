[![](https://risencrypto.github.io/images/Logo.png?raw=true)](https://risencrypto.github.io/)

# [Risen Crypto](https://risencrypto.github.io/)

Mathematical Cryptography, zkSNARKs

[Blog](https://risencrypto.github.io/) [About](https://risencrypto.github.io/about/) [Contact](https://risencrypto.github.io/contact/) [RSS](https://risencrypto.github.io/feed.xml)

# Sum-Check Protocol and Multilinear Extensions (MLEs)

- [The Sum-Check Protocol](https://risencrypto.github.io/Sumcheck/#the-sum-check-protocol)
- [The ♯♯SAT Problem](https://risencrypto.github.io/Sumcheck/#the-sharpsat-problem)
- [Extension Polynomials and Multilinear Extensions](https://risencrypto.github.io/Sumcheck/#extension-polynomials-and-multilinear-extensions)
  - [Optimisations](https://risencrypto.github.io/Sumcheck/#optimisations)
- [Encoding the Adjacency Matrix of a Graph](https://risencrypto.github.io/Sumcheck/#encoding-the-adjacency-matrix-of-a-graph)
- [Counting Triangles](https://risencrypto.github.io/Sumcheck/#counting-triangles)

### The Sum-Check Protocol

gg is a vv-variate polynomial defined over a Field FF.

u=∑x1∈{0,1}∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)−(EqI)u=∑x1∈{0,1}∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)−(EqI)

The above equation sums gg over all possible Boolean inputs. The Prover (PP) claims that the sum is uu. The Sum-Check protocol allows PP to convince a Verifier (VV) that she has computed the sum correctly.

The steps in the protocol

(1)(1)PP sends the value uu which she claims as the sum, to VV

(2)(2)PP then sends a univariate polynomial g1g1 in x1x1 to VV. PP claims it’s been created by summing gg in all variables except x1x1 i.e. computing the following sum keeping x1x1 free

g1(x1)=∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)g1(x1)=∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)

The right hand side above is the same expression as the right hand side in Expression (I) at the start except with the summation over x1x1 removed.

∑x1∈{0,1}∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)∑x1∈{0,1}∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)

EqIEqI results in a value in the left hand side because all variables were evaluated. Where g1g1 becomes a univariate polynomial in x1x1 instead of a value because x1x1 is not summed for all values in x1∈{0,1}x1∈{0,1}

(3)(3)VV checks if degree(g1)<=degree(g)degree(g1)<=degree(g). If not, VV rejects the proof.

(4)(4)VV then checks the one summation which was omitted by PP while creating g1g1. Doing that omitted summation should result in uu

i.e. check if ∑x1∈{0,1}g1(x1)=?u∑x1∈{0,1}g1(x1)=?u

So he computes g1(0)+g1(1)g1(0)+g1(1) & checks if it sums to uu.

If it does sum to uu, then VV can believe that gg also actually totals up to uu (which is the claim he is verifying) as long as he is sure that g1g1 was indeed constructed the way described above (in Step (2)(2))

Let’s assume there is a polynomial s1s1 which was actually created that way. i.e.

s1(x1)=∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)s1(x1)=∑x2∈{0,1}...∑xv∈{0,1}g(x1,x2,...,xv)

VV has to verify that g1=s1g1=s1. He can do this by picking a random element r1∈Fr1∈F & verifying if g1(r1)=?s1(r1)g1(r1)=?s1(r1) ( [Schwartz-Zippel Lemma](https://risencrypto.github.io/Kate/#the-schwartz-zippel-lemma))

The complexity of creating s1s1 & evaluating s1(r1)s1(r1) isn’t much lesser than evaluating the original gg polynomial for all inputs & checking if it equals uu. If VV were to do this, he could as well herself calculate gg instead of getting it done by PP.

So instead, VV checks if s1(r1)=?g1(r1)s1(r1)=?g1(r1) recursively as shown below.

(5)(5)VV choses a random element r1∈Fr1∈F & sends it to PP

(6)(6) Just like PP created g1g1 by summing gg in all variables except x1x1, now PP creates g2g2 by replacing x1x1 in gg with the value r1r1 & summing gg with x2x2 free/unevaluated i.e. x1=r1x1=r1, x2x2 free/unevaluated & but expanding all possible values of x3x3 to xvxv.

This creates a univariate polynomial g2(x2)g2(x2). PP sends g2g2 to VV

(7)(7)VV checks the degree of g2g2 & rejects if not correct.

(8)(8)g2g2 is built with x1=r1x1=r1 & x2x2 kept free. So VV needs to check if g1(x1=r1)=?g2(x2=0)+g2(x2=1)g1(x1=r1)=?g2(x2=0)+g2(x2=1)

i.e. check if g1(r1)=?g2(0)+g2(1)g1(r1)=?g2(0)+g2(1) & reject if not.

If it checks out, then all VV needs to do is verify that g2g2 indeed has been constructed correctly as described. So he continues the recursion as done below.

(9)(9) We have seen 2 rounds above already. Steps (5)(5) to (8)(8) are repeated for j=3j=3 to j=v−1j=v−1

- VV choses a random element rj−1∈Frj−1∈F & sends it to PP
- PP creates the univariate polynomial gj(xj)gj(xj) by keeping xjxj free and x1x1 to xj−1xj−1 fixed at r1r1 to rj−1rj−1 respectively

gj(xj)=∑xj+1∈{0,1}∑xj+2∈{0,1}...∑xv∈{0,1}g(r1,r2,...,rj−1,xj,xj+1,...,xv)gj(xj)=∑xj+1∈{0,1}∑xj+2∈{0,1}...∑xv∈{0,1}g(r1,r2,...,rj−1,xj,xj+1,...,xv)

- VV checks the degree & also if gj−1(rj−1)=?gj(0)+gj(1)gj−1(rj−1)=?gj(0)+gj(1)

(10)(10) In the last round, PP sends gv(xv)gv(xv) to VV. VV checks degree & if gv−1(rv−1)=?gv(0)+gv(1)gv−1(rv−1)=?gv(0)+gv(1)

(11)(11)VV then picks a random rv∈Frv∈F & evaluates g(r1,r2,…,rv)g(r1,r2,…,rv). VV also checks if gv(rv)=?g(r1,r2,…,rv)gv(rv)=?g(r1,r2,…,rv) & rejects if not.

If it checks out, then it means gv−1gv−1 is the same polynomial as sv−1sv−1 i.e. it has been constructed like the PP said he constructed it. Which in turn means gv−2gv−2 is the same as sv−2sv−2. As we keep unrolling, finally, it proves that g1g1 is equal to s1s1 which proves the original summation sums up to uu. This ends the proof.

### The ♯♯SAT Problem

We will run through the Sum-Check protocol using a ♯SAT♯SAT example

(x1&x2)&(x3\|x4)(x1&x2)&(x3\|x4)

In ♯SAT♯SAT you have to check for all possible different combinations of x1,x2,x3,x4x1,x2,x3,x4 & count how many combinations end up satisfying the above boolean circuit/formula i.e. evaluate to TRUE(1).

Any boolean formula can be arithmetized & converted into a function by replacing

- A&BA&B with A∗BA∗B
- A\|BA\|B with (A+B)−(A∗B)(A+B)−(A∗B)
- !A!A with 1−A1−A

Arithmetizing the above boolean circuit gives us
g(x1,x2,x3,x4)=(1−x1)x2((x3+x4)−x3x4)g(x1,x2,x3,x4)=(1−x1)x2((x3+x4)−x3x4)

We will walk through the protocol using a sagemath program.

∑x1∈{0,1}∑x2∈{0,1}∑x3∈{0,1}∑x4∈{0,1}g(x1,x2,x3,x4)∑x1∈{0,1}∑x2∈{0,1}∑x3∈{0,1}∑x4∈{0,1}g(x1,x2,x3,x4)

We will operate in a Finite Field F97F97. First PP evaluates gg for all possible values of x1,x2,x3,x4x1,x2,x3,x4 i.e.

```
F97 = GF(97)
R97.<x1,x2,x3,x4> = PolynomialRing(F97)
g = R97((1-x1)*x2*((x3+x4)-x3*x4))
sum = 0

for a1 in [0,1]:
    for a2 in [0,1]:
        for a3 in [0,1]:
            for a4 in [0,1]:
                sum = sum + g(a1,a2,a3,a4)
```

Sum evaluates to 33

PP creates a univariate polynomial in x1x1

```
g1 = R97(0)
for a2 in [0,1]:
    for a3 in [0,1]:
        for a4 in [0,1]:
           g1 = g1 + g(x1,a2,a3,a4)
```

If you print out g1g1,

g1(x1)=−3x1+3g1(x1)=−3x1+3

VV computes g1(x1=0)+g1(x1=1)g1(x1=0)+g1(x1=1) & checks if it evaluates to 3.

VV picks random number r1∈F97r1∈F97, r1=25r1=25 & sends it to PP

PP keeps x1=r1x1=r1 & creates the next univariate polynomial g2g2 keeping x2x2 free.

```
r1 = 25
g2 = R97(0)
for a3 in [0,1]:
    for a4 in [0,1]:
       g2 = g2 + g(r1,x2,a3,a4)
```

This creates g2(x2)=25x2g2(x2)=25x2 & is sent to VV

VV checks if g1(x1=r1)=?g2(x2=0)+g2(x2=1)g1(x1=r1)=?g2(x2=0)+g2(x2=1), which checks out - both LHS & RHS are 25.

Then,

```
r2 = 6
g3 = R97(0)
for a4 in [0,1]:
   g3 = g3 + g(r1,r2,x3,a4)
```

g3(x3)=−47x3−47g3(x3)=−47x3−47

g2(x2=r2)=g3(x3=0)+g3(x3=1)=−44g2(x2=r2)=g3(x3=0)+g3(x3=1)=−44 \- checks out.

```
r3 = 11
g4 = R97(g(r1,r2,r3,x4))
```

g4(x4)=−15x4−32g4(x4)=−15x4−32

VV picks r4=3r4=3

g4(x4=r4)=−3g4(x4=r4)=−3

g(r1,r2,r3,r4)=94g(r1,r2,r3,r4)=94 (same as −3mod97−3mod97)

So again checks out & the proof is done.

### Extension Polynomials and Multilinear Extensions

Let’s say we have a vector a=\[11,7,23,14\]a=\[11,7,23,14\] & we want to encode it into a univariate polynomial, one of the ways to do is to first consider these elements as y-coordinates of points corresponding to x∈\[0,1,2,3\]x∈\[0,1,2,3\] like we did in [QAP](https://risencrypto.github.io/R1CSQAP/#qap).

We start with a map f:{0,1,2,3}↦{11,7,23,14}f:{0,1,2,3}↦{11,7,23,14}
& we then use Lagrange Interpolation of the points \[(0,11),(1,7),(2,23),(3,14)\]\[(0,11),(1,7),(2,23),(3,14)\] to create the polynomial f~(x)=41x3+81x2+68x+11f~(x)=41x3+81x2+68x+11.

Though we were interested only in inputs of 0,1,2,30,1,2,3, the polynomial can take all inputs in F97F97
i.e.
f~(0)=11,f~(1)=7,f~(2)=23,f~(3)=14,f~(4)=32,f~(5)=32,…,f~(95)=65,f~(96)=80f~(0)=11,f~(1)=7,f~(2)=23,f~(3)=14,f~(4)=32,f~(5)=32,…,f~(95)=65,f~(96)=80

We started with a vector a=\[11,7,23,14\]a=\[11,7,23,14\] and encoded it into a polynomial whose domain can be considered as a much larger vector a′=\[11,7,23,14,32,32,…,65,80\]a′=\[11,7,23,14,32,32,…,65,80\]

This polynomial f~f~ maps F97↦F97F97↦F97

Hence the polynomial f~f~ is called an **extension** of the original map/function which mapped just those 4 elements.

If \|a\|\|a\| is much smaller than \|a′\|\|a′\| (i.e size of field is much larger), then it’s called a **low degree extension** (numercial examples in this post do not operate in a very large field, we operate in F97F97 for ease of understanding - so these examples don’t actually generate a low degree extension)

The above was a Univariate Polynomial. A Multivariate Polynomial which has a maximum degree of 11 in each of it’s variables is called as Multilinear Polynomial. For e.g. 2xy+3x+y2xy+3x+y is a multilinear polynomial but 2x2+3xy+52x2+3xy+5 isn’t multilinear.

For the univariate case, we considered functions/maps whose domain was {0,1,2,…,n−1}{0,1,2,…,n−1} which mapped to FF & we interpolated their extension. In the multivariate case, we will look at functions/maps whose domain is {0,1}v{0,1}v & map it to FF.

In the univariate case, if we had a vector of nn values which we wanted to encode in our univariate polynomial, we picked the input domain as {0,1,…,n−1}{0,1,…,n−1} \- like for a=\[11,7,23,14\]a=\[11,7,23,14\] of size 44, we used the input domain {0,1,2,3}{0,1,2,3}. In the multivariate case, for encoding a vector of size nn, we pick v=lognv=logn & use {0,1}v{0,1}v as the input domain. {0,1}v{0,1}v is called as the vv-dimensional Boolean Hypercube.

**Lagrange Interpolation of a Multilinear Extension (MLE) Polynomial**

Let’s encode the same vector {11,7,23,14}{11,7,23,14} into a multilinear polynomial.

Like before, we start with a map f:{0,1,2,3}↦{11,7,23,14}f:{0,1,2,3}↦{11,7,23,14}.

Our input domain size is 4, so our v=log(4)=2v=log(4)=2. So we use {0,1}2{0,1}2. This has 4 values - 0,1,2,30,1,2,3 \- expressing this in binary it becomes 00,01,10,1100,01,10,11. So each element of the input domain has 2 bits - i.e. so we need a bivariate polynomial f~(x1,x2)f~(x1,x2) where x1x1 will take the value of the first bit & x2x2 the 2nd bit. In general, x1x1 will take the value of the Most Significant Bit (MSB) & the last xvxv will be the LSB. So if we start with a vector of size nn, we will end up with a vv-variate polynomial where v=lognv=logn

For the interpolation, we first calculate the Multilinear Lagrange Basis Polynomials using this formula

Lw(x1,x2,...,xv)=∏i=1v(xiwi+(1−xi)(1−wi))−(EqII)Lw(x1,x2,...,xv)=∏i=1v(xiwi+(1−xi)(1−wi))−(EqII)

wiwi is the current bit - for e.g. for the input bitstring 1010, w1=1,w2=0w1=1,w2=0

Let’s create L00L00 corresponding to input 00 (bitstring 0000)

L00=∏2i=1(xiwi+(1−xi)(1−wi))L00=∏i=12(xiwi+(1−xi)(1−wi))

=(x1w1+(1−x1)(1−w1))⋅(x2w2+(1−x2)(1−w2))=(x1w1+(1−x1)(1−w1))⋅(x2w2+(1−x2)(1−w2))

Here, w1=0w1=0 & w2=0w2=0

L00=(1−x1).(1−x2)=1−x2−x1+x1x2L00=(1−x1).(1−x2)=1−x2−x1+x1x2

For L01L01, w1=0w1=0 & w2=1w2=1

L01=(1−x1).(x2)=x2−x1x2L01=(1−x1).(x2)=x2−x1x2

L10=x1−x1x2L10=x1−x1x2

L11=x1x2L11=x1x2

Using the Lagrange basis Polynomials, we can now calculate the Multilinear extension for our map f:{1,2,3,4}↦{11,7,23,14}f:{1,2,3,4}↦{11,7,23,14} using the below formula

f~(x1,x2,...,xv)=∑w∈{0,1}vf(w)⋅Lw(x1,x2,...,xv)−(EqIII)f~(x1,x2,...,xv)=∑w∈{0,1}vf(w)⋅Lw(x1,x2,...,xv)−(EqIII)

f~(x1,x2)=f(0)⋅L00+f(1)⋅L01+f(2)⋅L10+f(3)⋅L11f~(x1,x2)=f(0)⋅L00+f(1)⋅L01+f(2)⋅L10+f(3)⋅L11

=11⋅(1−x2−x1+x1x2)+7⋅(x2−x1x2)+23⋅(x1−x1x2)+14⋅(x1x2)=11⋅(1−x2−x1+x1x2)+7⋅(x2−x1x2)+23⋅(x1−x1x2)+14⋅(x1x2)

=11−11x2−11x1+11x1x2+7x2−7x1x2+23x1−23x1x2+14x1x2=11−11x2−11x1+11x1x2+7x2−7x1x2+23x1−23x1x2+14x1x2

f~(x1,x2)=11+12x1−4x2−5x1x2f~(x1,x2)=11+12x1−4x2−5x1x2

The map ff mapped only the input domain \[0,1,2,3\]\[0,1,2,3\] which is same as (x1,x2)=\[(0,0),(0,1),(1,0),(1,1)\](x1,x2)=\[(0,0),(0,1),(1,0),(1,1)\], but f~f~ can actually be evaluated for all possible x1&x2′s∈F97x1&x2′s∈F97 \- hence it is a multilinear extension of the map.

Here is a sage program to do the same thing which we did manually above

```
v = 2 #v-dimensional Boolean hypercube

fmap = [11, 7, 23, 14]

F97 = GF(97)
Lw = [] # Lagrange Basis
R97 = PolynomialRing(F97, v, [f"x{i}" for i in range(1, v+1)])

for i in range(2^v):
    b=Integer(i).digits(2, None, v) # pad(ZZ(i).binary(),v)
    g = R97(1)

    for j in range(v):
        xi = R97.gen(j)
        wi = b[v-1-j]
        g = g* (xi * wi + (1-xi)*(1-wi))

    Lw.append(g)

#MLE
f = R97(0)
for i in range(2^v):
    f = f+ fmap[i]*Lw[i]

print("MLE = " + str(f))
```

**Note:** Our gg polynomial, on which we run the Sum-Check Protocol need not be a multilinear polynomial. In some examples, you will see that it will be a product of multilinear polynomials & the product may not be multilinear. But in most practical cases, gg will have a degree at most of 22 or 33 in each of it’s variables.

#### Optimisations

The input to VV sent by PP is the values of the map f\[w\]∀w∈2vf\[w\]∀w∈2v. In the Sum-Check Protocol, the verifier has to only evaluate the MLE once at the end at a random point. So it may not be actually necessary for VV to first compute the MLE & then evaluate it. VV can instead directly evaluate the MLE without ever forming the polynomial.

Let us say that VV has to evaluate the MLE at random value r=r1,r2,…,rvr=r1,r2,…,rv, he can first compute Lagrange Basis Polynomials using EqIIEqII in the above section on MLE’s using x1=r1,x2=r2,…x1=r1,x2=r2,… & then use EqIIIEqIII directly with the ffs and the Lagrange Basis polynomials to evaluate the MLEs.

Let’s take an example.

PP sends VV the map f:{0,1}3↦Ff:{0,1}3↦F given by f(0,0,0)=1,f(0,1,0)=2,f(1,0,0)=3,f(1,1,0)=4,f(0,0,1)=5,f(0,1,1)=6,f(1,0,1)=7,f(1,1,1)=8f(0,0,0)=1,f(0,1,0)=2,f(1,0,0)=3,f(1,1,0)=4,f(0,0,1)=5,f(0,1,1)=6,f(1,0,1)=7,f(1,1,1)=8

VV has to evaluate the MLE f~f~ at a random point r=\[2,4,6\]r=\[2,4,6\] i.e. evaluate f~(2,4,6)f~(2,4,6)

```
v = 3
fmap = [1,2,3,4,5,6,7,8]
r = [2, 4, 6] # Random Point
F97 = GF(97)
sum = 0

for i in range(2^v):
    b=Integer(i).digits(2, None, v)
    # i'th Lagrange Base
    Lw = R97(1)

    for j in range(v):
        wi = b[v-1-j]

        # Eq II to form the Lagrange Basis
        Lw = Lw* (r[j] * wi + (1-r[j])*(1-wi))

    #Eq III
    sum = sum + fmap[i] * g
print(sum)
```

The above program will print 2323. So VV has successfully evaluated the MLE without actually forming the MLE.

If you look at the above program carefully, you will notice that it calculates does the multiplications (1−r1)⋅(1−r2)(1−r1)⋅(1−r2), (r1⋅r2)(r1⋅r2), (1−r1)⋅(r2)(1−r1)⋅(r2) multiple times. As vv becomes larger the number of multiplications which get repeated creates a huge inefficiency. We can use dynamic programming/memoization to optimise it (If you are not familiar with this concept, you can take a look at this video - [Dynamic Programming Tutorial with Fibonacci Sequence](https://www.youtube.com/watch?v=e0CAbRVYAWg) )

Let’s first assume v=1v=1 & calculate L0L0 & L1L1

L0=(1−r1)=−1L0=(1−r1)=−1

L1=(r1)=2L1=(r1)=2

Next with v=2v=2

L00=L0⋅(1−r2)=3L00=L0⋅(1−r2)=3

L01=L0⋅r2=−4L01=L0⋅r2=−4

L10=L1⋅(1−r2)=−6L10=L1⋅(1−r2)=−6

L11=L1⋅r2=8L11=L1⋅r2=8

With v=3v=3

L000=L00⋅(1−r3)=−15L000=L00⋅(1−r3)=−15

L001=L00⋅r3=18L001=L00⋅r3=18

L010=L01⋅(1−r3)=20L010=L01⋅(1−r3)=20

L011=L01⋅r3=−4⋅6=−24L011=L01⋅r3=−4⋅6=−24

L100=L10⋅(1−r3)=−4⋅(−6)=30L100=L10⋅(1−r3)=−4⋅(−6)=30

L101=L10⋅r3=−6⋅6=−36L101=L10⋅r3=−6⋅6=−36

L110=L11⋅(1−r3)=8⋅−5=−40L110=L11⋅(1−r3)=8⋅−5=−40

L111=L11⋅(r3)=8⋅6=48L111=L11⋅(r3)=8⋅6=48

Now we can evaluate the f~(2,4,6)f~(2,4,6) as

L000⋅f(0)+L001⋅f(1)+L010⋅f(2)+L011⋅f(3)+L100⋅f(4)+L101⋅f(5)+L110⋅f(6)+L111⋅f(7)L000⋅f(0)+L001⋅f(1)+L010⋅f(2)+L011⋅f(3)+L100⋅f(4)+L101⋅f(5)+L110⋅f(6)+L111⋅f(7)

which evaluates to 23

The same thing in a program

```
v = 3
fmap = [1,2,3,4,5,6,7,8]
r = [2, 4, 6]
F97 = GF(97)
Lw = [[1-r[0],r[0]]]

for j in range(1,v):
    Lwt = [F97(1)]*2^(j+1)

    for i in range(2^(j+1)):
        b = f'{i:b}'.zfill(j+1)
        lsb = b[j:j+1]
        rest =  b[0:j]

        if (lsb == "0"):
            Lwt[i] = R97(Lw[j-1][Integer('0b' + rest)]* (1-r[j]))
        else:
            Lwt[i] = R97(Lw[j-1][Integer('0b' + rest)]* r[j])

    Lw.append(Lwt)

# Compute f(2,4,6)
sum = 0
for i in range(2^v):
    sum = sum + fmap[i]*Lw[v-1][i]

print(sum)
```

### Encoding the Adjacency Matrix of a Graph

let’s consider a Graph G=(U,E)G=(U,E) where EE the set of kk edges & UU is the set of nn vertices numbered from 00 to n−1n−1 i.e. U={0,1,2,…,n−1}U={0,1,2,…,n−1}.

The Adjacency Matrix for this graph is a square matrix AA of size n×nn×n where we interpret the row & column numbers of elements as vertex numbers. If there is an edge between vertex ii & vertex jj the Matrix Element Aij=1Aij=1, else 00.

![](https://risencrypto.github.io/images/Graph.png?raw=true)

Lets take this graph.The adjacency matrix will be

Row0Row1Row2Row3Col00111Col11011Col21101Col31110Col0Col1Col2Col3Row00111Row11011Row21101Row31110

The adjacency matrix is a map f:(row,col)↦{0,1}f:(row,col)↦{0,1}

Since maximum row or column number is 44, it can each be represented by log(n)log(n) bits i.e.

f:{0,1}2×{0,1}2↦{0,1}f:{0,1}2×{0,1}2↦{0,1}

We will interpolate this to an MLE f~(row,col)f~(row,col). As said earlier, each row number & column number will take two bits to represent.

i.e. f~((r0,r1),(c0,c1))f~((r0,r1),(c0,c1)), where r0r0 & r1r1 are the bits of the row number. This is f~(r0,r1,c0,c1)f~(r0,r1,c0,c1) \- i.e. the MLE takes 4 parameters which can be represented by the 4 variables x1,x2,x3,x4x1,x2,x3,x4.

We will use our earlier program to interpolate the MLE with the following values of fmapfmap & vv

fmap=\[0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0\]fmap=\[0,1,1,1,1,0,1,1,1,1,0,1,1,1,1,0\]

v=4v=4 (2 bits each for row & column number)

Running the program, we get

f~=−4x1x2x3x4+2x1x2x3+2x1x2x4+2x1x3x4+2x2x3x4−x1x2−2x1x3−x2x3−x1x4−2x2x4−x3x4+x1+x2+x3+x4f~=−4x1x2x3x4+2x1x2x3+2x1x2x4+2x1x3x4+2x2x3x4−x1x2−2x1x3−x2x3−x1x4−2x2x4−x3x4+x1+x2+x3+x4

**Caveats in the above example**

A graph with 44 nodes will have an Adjacency Matrix of size 4×44×4 & the row & column numbers will go from 00 to 33, so they can be represented in 22 bits each, so any element in the matrix can be represented as (row, column) with 4 bits which works fine. But it’s not always so seamless.

![](https://risencrypto.github.io/images/Graph2.png?raw=true)

Let’s take for example the Adjacency Matrix for this 3 vertex graph. It’s a 3×33×3 matrix, so row, col numbers are from 00 to 22, this still needs 22 bits. So totally 44 bits will be needed to represent each element. So we need to convert the 3×33×3 Adjacency matrix into a 4×44×4 matrix by setting the extra elements as 00 before we interpolate it into an MLE.

Below you can see how the matrix can be built up into a 4×44×4 matrix. The elements in yellow are extra elements we added.

![Built up AdjacencyMatrix](https://risencrypto.github.io/images/AM.png?raw=true)

In general, if you need vv bits to represent each element & interpolate a vv-variate polynomial, then you will need to build up the matrix to have 2v2v elements i.e. a n×nn×n matrix where n=(√2v)n=(2v)

### Counting Triangles

Now, we will use our MLE to count the number of triangles in our 4 vertex graph (Note this is our first graph - the 4-vertex one & not the 3-vertex graph we discussed after that). We have already computed the MLE for this.

f~=−4x1x2x3x4+2x1x2x3+2x1x2x4+2x1x3x4+2x2x3x4−x1x2−2x1x3−x2x3−x1x4−2x2x4−x3x4+x1+x2+x3+x4f~=−4x1x2x3x4+2x1x2x3+2x1x2x4+2x1x3x4+2x2x3x4−x1x2−2x1x3−x2x3−x1x4−2x2x4−x3x4+x1+x2+x3+x4

Consider any 3 vertices out of the 4 - let’s call them a,b,ca,b,c. These 3 vertices will form a triangle if & only if there is an edge between aa & bb, and an edge between bb & cc & another between cc & aa. Which means, only if all three of f~(a,b),f~(b,c),f~(c,a)f~(a,b),f~(b,c),f~(c,a) evaluate to 1 - i.e. only if

f~(a,b)⋅f~(b,c)⋅f~(c,a)=1f~(a,b)⋅f~(b,c)⋅f~(c,a)=1

Even if one of the two vertices out of the three doesn’t have an edge between them, then that evaluation of f~f~ with those edges as params will be 00 i.e. the above multiplication will result in 00. So the above equation can be used to count the number of triangles in a graph. However, the triangle formed by 33 edges (ab)(bc)(ca)(ab)(bc)(ca) is the same as that formed by (ba)(cb)(ac)(ba)(cb)(ac) \- each triangle can be represented in 6 different ways by f~f~. So to get the count of the total number of triangles in the graph, we have to divide by 66.

16×∑a∈{0,1}2∑b∈{0,1}2∑c∈{0,1}2f~(a,b)⋅f~(b,c)⋅f~(c,a)16×∑a∈{0,1}2∑b∈{0,1}2∑c∈{0,1}2f~(a,b)⋅f~(b,c)⋅f~(c,a)

When PP computes the above, she will get the answer 2424. And 16×24=416×24=4

So count of triangles in our graph is 44 which can also be verified visually for a small graph like our 4-vertex graph.

PP proves this to VV using the Sum-Check protocol. Since we have already gone through all the steps of the protocol with other examples, I am not discussing each step here but only ones where we do things differently.

- PP doesn’t need to multiply the 3 f~f~’s to create a gg before evaluating it or running the Sum-Check protocol. Actually multiplying the f~f~’s makes it more complicated. She can instead evaluate the 3 f~f~’s & then multiply the evaluations to get 2424 which is then divided by 66 to get 44. For e.g.

```
sum = 0
for a1 in range(2):
  for a2 in range(2):
    for a3 in range(2):
      for a4 in range(2):
        for a5 in range(2):
          for a6 in range(2):
             sum = sum + f(a1, a2, a3, a4) * f(a3, a4, a5, a6) * f(a5, a6, a1, a2)

print(sum)
```

- In step 22, PP has to send the univariate polynomial g1g1 to VV. Again, this can be done without the verifier first creating a gg.

```
g1 = R97(0)
for a2 in range(2):
  for a3 in range(2):
    for a4 in range(2):
      for a5 in range(2):
        for a6 in range(2):
          g1 = g1 + f(x1,a2,a3,a4) * f(a3,a4,a5,a6) * f(a5,a6,x1,a2)
```

This gives us g1=−4x12+4x1+12g1=−4x12+4x1+12

But there is an even better way to do this. g1g1 is a polynomial of at most degree 22, so instead of creating g1g1 & sending it, PP can instead send evaluations of g1g1 at 3 points - i.e. values of g1(0)g1(0), g1(1)g1(1), g1(2)g1(2) & VV can recreate g1g1 himself at his end using Lagrange Interpolation for univariate polynomials.

```
a1 = [0,1,2]
sum = [0,0,0]

for i in range(len(a1)):
  for a2 in range(2):
    for a3 in range(2):
      for a4 in range(2):
        for a5 in range(2):
          for a6 in range(2):
            sum[i] = sum[i] + f(a1[i],a2,a3,a4) * f(a3,a4,a5,a6) * f(a5,a6,a1[i],a2)

print(sum)
```

This prints out \[12,12,4\]\[12,12,4\] \- i.e. g1(0)=12,g1(1)=12,g1(2)=4g1(0)=12,g1(1)=12,g1(2)=4

When VV uses univariate Lagrange Interoplation, then he will get the same g1=−4x12+4x1+12g1=−4x12+4x1+12

In all the steps where PP has to send a univariate gigi, she instead sends the appropriate number of evaluations instead.

Note that the above program has many calculations which are redone repeatedly & hence can be optimized further by dynamic programming.

In the final step of the protocol, VV has to compute f~(r1,r2)⋅f~(r2,r3)⋅f~(r3,r1)f~(r1,r2)⋅f~(r2,r3)⋅f~(r3,r1) where each riris is a vector of random values - as seen earlier each of the 3 are vectors of size logn=vlog⁡n=v. We pick r1,r2,r3∈Flogn×Flogn×Flognr1,r2,r3∈Flog⁡n×Flog⁡n×Flog⁡n. In our case, logn=v=2log⁡n=v=2, so VV has to pick 6 random numbers say r11,r12,r21,r22,r31,r32r11,r12,r21,r22,r31,r32 \- i.e. r1r1 is the vector {r11,r12}{r11,r12}, r2r2 the second two and so on. VV then computes f~(r11,r12,r21,r22)⋅f~(r21,r22,r31,r32)⋅f~(r31,r32,r11,r12)f~(r11,r12,r21,r22)⋅f~(r21,r22,r31,r32)⋅f~(r31,r32,r11,r12) which can be done in time linear to the size of the input Matrix.

**This post is based on Justin Thaler’s Book [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html)**

![Badge](https://hitscounter.dev/api/hit?url=https%3A%2F%2Frisencrypto.github.io%2FSumcheck%2F&label=Visitors&icon=github&color=%23198754&message=&style=flat&tz=Asia%2FCalcutta)

Written on January 31, 2024


- Share this:

- [Reddit](http://www.reddit.com/submit?url=https://risencrypto.github.io/Sumcheck/&title=Sum-Check%20Protocol%20and%20Multilinear%20Extensions%20(MLEs))
- [Y Combinator](http://news.ycombinator.com/submitlink?u=https://risencrypto.github.io/Sumcheck/&t=Sum-Check%20Protocol%20and%20Multilinear%20Extensions%20(MLEs))
- [Twitter](https://twitter.com/intent/tweet?url=https://risencrypto.github.io/Sumcheck/&text=Sum-Check%20Protocol%20and%20Multilinear%20Extensions%20(MLEs))
- [LinkedIn](https://www.linkedin.com/shareArticle?mini=true&url=https://risencrypto.github.io/Sumcheck/&title=Sum-Check%20Protocol%20and%20Multilinear%20Extensions%20(MLEs))

utterances

# 0 Comments _\- powered by_ _[utteranc.es](https://utteranc.es/)_

![@anonymous](<Base64-Image-Removed>)

Write

Preview


Nothing to preview


[Styling with Markdown is supported](https://guides.github.com/features/mastering-markdown/) [Sign in with GitHub](https://api.utteranc.es/authorize?redirect_uri=https%3A%2F%2Frisencrypto.github.io%2FSumcheck%2F)