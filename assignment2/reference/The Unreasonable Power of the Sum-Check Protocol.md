[![logo](https://zkproof.org/wp-content/uploads/2019/12/ZKProof-Final-Logo-horizontal-08-e1575314521465.png)](https://zkproof.org/)

- [Home](https://zkproof.org/ "Home")
- [About](https://zkproof.org/about/ "About")
- [Events](https://zkproof.org/events/ "Events")
- [Resources](https://docs.zkproof.org/ "Resources")
- [Forum](https://community.zkproof.org/ "Forum")
- [Gallery](https://zkproof.org/gallery/ "Gallery")
- [Blog](https://zkproof.org/blog/ "Blog")
- [ZKProof 8 in Rome](https://zkproof.org/events/zkproof-8-rome/ "ZKProof 8 in Rome")

# The Unreasonable Power of the Sum-Check Protocol

March 16, 2020

\|In [The Art of Zero Knowledge](https://zkproof.org/category/the-art-of-zero-knowledge/ "View all posts in The Art of Zero Knowledge")

\|By [Justin Thaler](https://zkproof.org/author/justinthaler/)

## Introduction

When designing an efficient interactive proof system, there is only one hammer you need to have in your toolbox: the _sum-check protocol_ of [Lund, Fortnow, Karloff, and Nisan](https://dl.acm.org/doi/10.1145/146585.146605). The power of this protocol seems to be a bit under-appreciated in the ZKProofs community. I speculate that there are two reasons for this. The first is that the protocol inherently leads to proofs of at least logarithmic length, which means that in applications where super short proofs are really important—say, because these proofs need to be stored for all eternity on a blockchain—other techniques may be preferable (e.g., protocols derived from the work of [Gennaro, Gentry, Parno, and Raykova](https://eprint.iacr.org/2012/215), which tend to have constant proof length).

Second, the sum-check protocol by itself is not zero-knowledge nor “succinct for NP statements”. This means that, for NP statements, the proof length achieved by the sum-check protocol is not sublinear in the size of the NP witness. Note that there is [strong evidence](https://www.math.ias.edu/~avi/PUBLICATIONS/MYPAPERS/ODED/IP/FINAL/paper.pdf) that _no_ interactive proof can be succinct for NP statements. This is in contrast to argument systems (which unlike interactive proofs, are only computationally sound).

However, in the last few years, researchers have figured out how to combine the sum-check protocol with cryptographic commitments to obtain arguments that are both zero-knowledge and succinct for NP statements. This has led to zk-SNARKs with state of the art performance (e.g., [Hyrax](https://eprint.iacr.org/2017/1132), [zk-vSQL](https://eprint.iacr.org/2017/1146), [Libra](https://eprint.iacr.org/2019/317), [Virgo](https://eprint.iacr.org/2019/1482.pdf), [Spartan](https://eprint.iacr.org/2019/550)). So if you are interested in zk-SNARKs and satisfied with proofs of logarithmic length, then I urge you to learn about the sum-check protocol and how to use it.

To this end, the goal of this blog post is to describe the problem solved by the sum-check protocol and why it is so useful. For readers interested in learning more, I have posted a [more detailed (and unfortunately more technical) exposition here](http://people.cs.georgetown.edu/jthaler/blogpost.pdf).

This entire post will be framed in the context of interactive proofs (IPs). This means that the goal is for a verifier 𝑉 to offload an expensive computation to an untrusted prover 𝑃, while achieving work-saving for the verifier. We want the verifier to run in time linear in the input size, while keeping the proof short (logarithmic size) and the prover efficient. In a future post, I will explain how to combine the ideas described here with cryptographic commitments to get state of the art zk-SNARKs.

## The Sum-Check Protocol

Suppose we are given a 𝑣-variate polynomial 𝑔 defined over a finite field 𝐅. Let us further assume that 𝑔 has degree at most 2 in each variable, as this will be the case in all of the applications in both this post and [its more detailed version](http://people.cs.georgetown.edu/jthaler/blogpost.pdf). The purpose of the sum-check protocol is to compute the sum:

𝐻:=∑𝑏1∈{0,1}∑𝑏2∈{0,1}…∑𝑏𝑣∈{0,1}𝑔⁡(𝑏1,…,𝑏𝑣)(1)

Summing up the evaluations of a polynomial over all Boolean inputs may seem like a contrived task with limited practical utility. But to the contrary,

later sections of this post will show that many natural problems can be directly cast as an instance of Equation (1).

To keep this post as nontechnical as possible, I will say only a little about how the sum-check protocol actually works; see the [more detailed post](http://people.cs.georgetown.edu/jthaler/blogpost.pdf) for a complete description of the protocol and why it is sound. The protocol consists of 𝑣 rounds, one for each variable of 𝑔. In each round 𝑖, the prover sends to the verifier a degree 2 univariate polynomial 𝑔𝑖 (𝑔𝑖 can always be specified with just 3 field elements, by either sending its coefficients, or its evaluations at 3 designated inputs in 𝐅). The verifier performs some simple randomized consistency checks on each message 𝑔𝑖; these checks involve evaluating 𝑔𝑖 at a handful of inputs and checking that these evaluations are consistent with previous messages sent by the prover. The verifier can process each message sent by the prover in 𝑂⁡(1) time, and at the very end of the protocol the verifier also needs to evaluate 𝑔 at single point. Throughout, we assume any addition or multiplication operation in 𝐅 takes 𝑂⁡(1) time.

### What does the verifier gain by using the sum-check protocol?

The verifier could clearly compute 𝐻 via Equation (1) on her own by evaluating 𝑔 at 2𝑣 inputs (namely, all inputs in {0,1}𝑣), but we are thinking of 2𝑣 as an unacceptably large runtime for the verifier. Using the sum-check protocol, the verifier’s runtime is

𝑂⁡(𝑣+\[thecosttoevaluate⁢𝑔atasingleinputin𝐅𝑣\])

This is much better than the 2𝑣 evaluations of 𝑔 required to compute 𝐻 unassisted. It also turns out that the prover in the sum-check protocol can compute all of its prescribed messages by evaluating 𝑔 at 𝑂⁡(2𝑣) inputs in 𝐅𝑣. This is only a constant factor more than what is required simply to compute 𝐻 without proving correctness. The soundness error of the sum-check protocol is 𝑂⁢(𝑣/\|𝐅\|). As long as 𝑔 is defined over a field of size significantly greater than 𝑣, this error is very small.

## Applications

At this point in the post, we have our hammer in hand: the sum-check protocol, which allows a verifier to offload the computation expressions of the form of Equation (1) to an untrusted prover. However, wielding this hammer to solve problems people care about can require a good deal of cleverness. The goal of the rest of this post is give a flavor of how this typically works.

The general challenge is the following: suppose the verifier has an input 𝑥 and asks the prover to compute some function 𝐹 of 𝑥. To apply the sum-check protocol to compute 𝐹, we need to be able to express 𝐹⁡(𝑥) as an instance of Equation (1). This means that we need to identify some 𝑣-variate polynomial 𝑔 of degree 2 in each variable such that 𝐹⁡(𝑥) can be inferred from the sum of 𝑔’s values over all inputs in {0,1}𝑣. Moreover, the verifier needs to be able to evaluate 𝑔⁡(𝑟) at any desired input 𝑟∈𝐅𝑣 in linear time.

I will explain one illustrative example of this paradigm, in which the input 𝑥 is the adjacency matrix of a graph, and 𝐹⁡(𝑥) is the number of triangles in that graph. To accomplish this, I have no choice but to introduce one technical notion, called multilinear extensions, defined in the lemma below. To avoid unnecessary details, I do not prove the lemma, but it follows readily from [Lagrange interpolation](https://en.wikipedia.org/wiki/Lagrange_polynomial).

### Multilinear Extension Lemma

Let 𝑓:{0,1}𝑛→𝐅. Then there is a unique multilinear polynomial ˜𝑓 over 𝐅 such that ˜𝑓⁡(𝑥)=𝑓⁡(𝑥) for all 𝑥∈{0,1}𝑛. Here, a polynomial is said to be multilinear if it has degree at most 1 in each variable. ˜𝑓 is called the _multilinear extension_ (MLE) of 𝑓. Given as input a list of all 2𝑛 evaluations of 𝑓, and an arbitrary point 𝑟∈𝐅𝑛, there is an algorithm that can evaluate ˜𝑓⁡(𝑟) in 𝑂⁡(2𝑛) time.

### An Application: An IP for Counting Triangles

Let 𝐺 be an undirected graph on 𝑛 vertices with edge set 𝐸. Let 𝐴∈{0,1}𝑛×𝑛 be the adjacency matrix of 𝐺, i.e., 𝐴𝑖,𝑗=1 if and only if (𝑖,𝑗)∈𝐸. In the counting triangles problem, the input is the adjacency matrix 𝐴, and the goal is to determine the number of vertex triples (𝑖,𝑗,𝑘) that are all connected to each other by edges.

At first blush, it is totally unclear how to express the number of triangles in 𝐺 as the sum of the evaluations of a degree-2 polynomial 𝑔 over all inputs in {0,1}𝑣. After all, the counting triangles problem itself makes no reference to any low-degree polynomial 𝑔, so where will 𝑔 come from? This is where multilinear extensions come to the rescue.

For it to make sense to talk about multilinear extensions, we need to view the adjacency matrix 𝐴 not as a matrix, but rather as a function 𝑓𝐴 mapping {0,1}log⁡𝑛×{0,1}log⁡𝑛 to {0,1}. The natural way to do this is to define 𝑓𝐴⁡(𝑥,𝑦) so that it interprets 𝑥 and 𝑦 as the binary representations of some integers 𝑖 and 𝑗 between 1 and 𝑛, and outputs 𝐴𝑖,𝑗.

Then the number of triangles in 𝐺 is simply: Δ:=16⁢∑𝑥,𝑦,𝑧∈{0,1}log⁡𝑛𝑓𝐴⁡(𝑥,𝑦)⋅𝑓𝐴⁡(𝑦,𝑧)⋅𝑓𝐴⁡(𝑥,𝑧)(2)

To see that this equality is true, observe that the term for 𝑥,𝑦,𝑧 in the above sum is 1 if edges (𝑥,𝑦), (𝑦,𝑧), and (𝑥,𝑧) all appear in 𝐺, and is 0 otherwise. The factor 1/6 comes in because the sum over _unordered_ node triples (𝑖,𝑗,𝑘) counts each triangle 6 times, once for each permutation of 𝑖, 𝑗, and 𝑘.

Let 𝐅 be a finite field of size 𝑝≥𝑛3, where 𝑝 is a prime, and let us view all entries of 𝐴 as elements of 𝐅. Here, we are choosing 𝑝 large enough so that 6⁢Δ is guaranteed to be in {0,1,…,𝑝}. This ensures that, if we associate elements of 𝐅 with integers in {0,1,…,𝑝} in the natural way, then Equation (2) holds even when all additions and multiplications are done in 𝐅 rather than over the integers. (Choosing a large field to work over has the added benefit of ensuring good soundness error, as the soundness error of the sum-check protocol decreases linearly with field size.)

At last we are ready to describe the polynomial 𝑔 to which we will apply the sum-check protocol to compute 6⁢Δ. Recalling that ˜𝑓𝐴 is the MLE of 𝑓𝐴 over 𝐅, define the (3⁢log⁡𝑛)-variate polynomial 𝑔 to be: 𝑔⁡(𝑋,𝑌,𝑍)=˜𝑓𝐴⁡(𝑋,𝑌)⋅˜𝑓𝐴⁡(𝑌,𝑍)⋅˜𝑓𝐴⁡(𝑋,𝑍)

It is easy to see that 6⁢Δ=∑𝑥,𝑦,𝑧∈{0,1}log⁡𝑛𝑔⁡(𝑥,𝑦,𝑧), so applying the sum-check protocol to 𝑔 yields an IP computing 6⁢Δ. This IP requires 3⁢log⁡𝑛 rounds, with the prover sending 3 field elements in each round. The verifier’s runtime is dominated by the time required to evaluate 𝑔 at a single input (𝑟1,𝑟2,𝑟3)∈𝐅3⁢log⁡𝑛, for which it suffices to evaluate ˜𝑓𝐴 at the three inputs (𝑟1,𝑟2), (𝑟2,𝑟3), and (𝑟1,𝑟3). This can be done in 𝑂⁡(𝑛2) time using the Multilinear Extension Lemma. This is much faster than the fastest known algorithm for counting triangles, which runs in matrix multiplication time (superlinear in the input size).

It turns out that the prover in this IP can compute all of its prescribed messages in 𝑂⁡(𝑛3) time. This is not obvious, and for brevity, I’ll omit the details of how to accomplish this. Note that this runtime for the prover matches that of the the naive algorithm for counting triangles that iterates over all triples of vertices in 𝐺 and checks if they are all connected to each other.

### More Applications

Hopefully the above gives a sense of how problems that people care about can be expressed as instances of Equation (1) in non-obvious ways. The general paradigm works as follows. An input 𝑥 of length 𝑛 is viewed as a function 𝑓𝑥 mapping {0,1}log⁡𝑛 to some field 𝐅. And then the multilinear extension ˜𝑓𝑥 of 𝑓𝑥 is used in some way to construct a low-degree polynomial 𝑔 such that, as per Equation (1), the desired answer equals the sum of the evaluations of 𝑔 over the Boolean hypercube.

The [full version](http://people.cs.georgetown.edu/jthaler/blogpost.pdf) of this post covers some additional examples of this paradigm. In the hopes of enticing you to check it out, here is a summary of the examples covered there.

First, it gives a more sophisticated IP for counting triangles in which the prover is much more efficient than the above. Specifically, the prover runs the best-known algorithm to solve the triangles problem, and then does a low-order amount of extra work to prove the answer is correct. I don’t know of any other IPs or argument systems that achieve this super-efficiency for the prover while keeping the proof length sublinear in the input size.

Second, it gives a similarly super-efficient IP for matrix-powering. Given any 𝑛×𝑛 matrix 𝐴 over field 𝐅, this IP is capable of computing any desired entry of the powered matrix 𝐴𝑘. The number of rounds and communication cost of the IP are 𝑂⁢(log⁡(𝑘)⋅log⁡𝑛), and the verifier’s runtime is 𝑂⁢(𝑛2+log⁡(𝑘)⋅log⁡𝑛).

Finally, it uses this matrix-powering protocol to re-prove the following important result of [Goldwasser, Kalai, and Rothblum](https://www.microsoft.com/en-us/research/wp-content/uploads/2008/01/GoldwasserKR08a.pdf) (GKR): all problems solvable in logarithmic space have an IP with a quasilinear-time verifier, polynomial time prover, and polylogarithmic proof length. The basic idea of the proof is that executing any Turing Machine 𝑀 that uses 𝑠 bits of space can be reduced to the problem of computing a single entry of 𝐴2𝑠 for a certain matrix 𝐴 (𝐴 is in fact the [configuration graph](https://en.wikipedia.org/wiki/Configuration_graph) of 𝑀). So one can just apply the matrix-powering IP to 𝐴 to determine the output of 𝑀. While 𝐴 is a huge matrix (it has at least 2𝑠 rows and columns), configuration graphs have a ton of structure, and this enables the verifier to evaluate ˜𝑓𝐴 at a single input in 𝑂⁢(𝑠⋅𝑛) time. If 𝑠 is logarithmic in the input size, then this means that the verifier in the IP runs in 𝑂⁡(𝑛⁢log⁡𝑛) time.

The original paper of GKR proved the same result by constructing an arithmetic circuit for computing 𝐴2𝑠 and then applying a sophisticated IP for arithmetic circuit evaluation to that circuit. The approach described above is simpler, in that it directly applies a simple IP for matrix-powering, rather than a more complicated IP for the general circuit-evaluation problem.

* * *

![](https://secure.gravatar.com/avatar/bedec8777fe43c552036bc6c6af55e3a78a57a1d5c62cc6af8edff0732453cc0?s=240&d=identicon&r=g)

##### [Justin Thaler](https://zkproof.org/author/justinthaler/ "Justin Thaler post page")

[All author posts](https://zkproof.org/author/justinthaler/ "Justin Thaler post page")

##### Related Posts

[![](https://zkproof.org/wp-content/uploads/2023/02/Sangria-Square-1-uai-886x443.png)](https://zkproof.org/2023/02/21/sangria-a-folding-scheme-for-plonk/)

February 21, 2023

### [Sangria: a Folding Scheme for PLONK](https://zkproof.org/2023/02/21/sangria-a-folding-scheme-for-plonk/)

In this technical note we present…

* * *

[![](https://secure.gravatar.com/avatar/ceb909d0a4d67fea894e68f21da71c2e85225d07fc95d5898a18fc44e3e70e9e?s=40&d=identicon&r=g)by ZKProof Standards](https://zkproof.org/author/contact70d66e844e/)

[![](https://zkproof.org/wp-content/uploads/2021/11/VDF-blogpost-NOV21_imageonly-uai-1032x516.jpg)](https://zkproof.org/2021/11/24/practical-snark-based-vdf/)

November 24, 2021

### [Practical SNARK-based VDF](https://zkproof.org/2021/11/24/practical-snark-based-vdf/)

Protocol Labs, the Ethereum Foundation,…

* * *

[![](https://secure.gravatar.com/avatar/c7c48ad4ca01ec4b6158a41aef794dd173ad3a4d6b9357b2f6ee1d8ab3cfaeef?s=40&d=identicon&r=g)by Jonathan Gross](https://zkproof.org/author/jpgross3/)

[![](https://zkproof.org/wp-content/uploads/2021/09/ZBF-darlin-blogpost-SEP21_imageonly-uai-1032x516.jpg)](https://zkproof.org/2021/09/29/darlin-recursive-proofs/)

September 29, 2021

### [Darlin: Proof-carrying data based on Marlin](https://zkproof.org/2021/09/29/darlin-recursive-proofs/)

In this blog post, we describe Darlin,…

* * *

[![](https://secure.gravatar.com/avatar/74a42cbaa5dec474a2e69d0a4e679b36e88a65812b9ed766cd0dfe68384622d4?s=40&d=identicon&r=g)by Ulrich Haböck](https://zkproof.org/author/ulrichhorizenlabsio/)

### Leave a Reply[Cancel reply](https://zkproof.org/2020/03/16/sum-checkprotocol/\#respond)

Post a Comment

Log in or provide your name and email to leave a comment.

Email me new posts

InstantlyDailyWeekly

Email me new comments

Save my name, email, and website in this browser for the next time I comment.

Comment

Δ

This site uses Akismet to reduce spam. [Learn how your comment data is processed.](https://akismet.com/privacy/)

- [Prev](https://zkproof.org/2020/02/27/zkp-set-membership/)
- [Next](https://zkproof.org/2020/06/08/recursive-snarks/)

#### Get notified of upcoming events & updates

Δ

#### Get notified of upcoming events & updates

Leave this field empty if you're human:

![](https://zkproof.org/wp-content/uploads/2019/01/Logo-Concepts-18-e1579319446772-uai-258x221.png)

#### Quick Links

[About](https://zkproof.org/about/)

[Events](https://zkproof.org/events/)

[Gallery](https://zkproof.org/gallery/)

[Resources](https://docs.zkproof.org/)

[Forum](http://community.zkproof.org/)

#### Get Connected

contact@zkproof.org

© 2026 ZKProof Standards. All rights reserved

### Privacy Preference Center

#### Privacy Preferences

## Discover more from ZKProof Standards

Subscribe now to keep reading and get access to the full archive.

Type your email…

Subscribe

[Continue reading](https://zkproof.org/2020/03/16/sum-checkprotocol/#)