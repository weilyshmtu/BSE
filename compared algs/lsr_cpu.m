function [C, time_used] = lsr_cpu(lambda,opts)
tic
I = eye(opts.n);
XtX = opts.XtX;
C = (XtX + lambda*I)\XtX;
C = C - diag(diag(C));
time_used = toc;