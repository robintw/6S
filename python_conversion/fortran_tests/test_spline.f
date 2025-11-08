      program test_spline
c     Test program for SPLINE and SPLINT subroutines
      parameter (nmax=100)
      integer n, i
      real x(nmax), y(nmax), y2(nmax)
      real yp1, ypn, xtest, ytest

c     Test data: y = sin(x) for x in [0, 2*pi]
      n = 10
      do 10 i=1,n
        x(i) = (i-1) * 6.283185307 / (n-1)
        y(i) = sin(x(i))
10    continue

c     Natural spline (2nd derivatives = 0 at endpoints)
      yp1 = 1.0e30
      ypn = 1.0e30

      call spline(x, y, n, yp1, ypn, y2)

c     Output spline coefficients
      write(*,*) 'Spline coefficients (y2):'
      do 20 i=1,n
        write(*,'(I3,2F15.8)') i, x(i), y2(i)
20    continue

c     Test interpolation at a few points
      write(*,*) ''
      write(*,*) 'Interpolation test:'
      do 30 i=1,5
        xtest = (i-1) * 6.283185307 / 4.0
        call splint(x, y, y2, n, xtest, ytest)
        write(*,'(A,F10.6,A,F15.8,A,F15.8)')
     &    'x=', xtest, ' y_spline=', ytest,
     &    ' y_exact=', sin(xtest)
30    continue

      end

c     Include the actual subroutines
      include 'spline_funcs.f'
