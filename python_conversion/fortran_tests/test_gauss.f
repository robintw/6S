      program test_gauss
c     Test program for GAUSS subroutine (Gauss-Legendre quadrature)
      integer n, i
      parameter (nmax=100)
      real x(nmax), w(nmax)
      real x1, x2
      double precision sum, exact, error

c     Test: integrate x^2 from 0 to 1
c     Exact answer = 1/3 = 0.333333...
      n = 5
      x1 = 0.0
      x2 = 1.0

      call gauss(x1, x2, x, w, n)

c     Output abscissas and weights
      write(*,*) 'Gauss-Legendre quadrature points and weights:'
      write(*,'(A,I3)') 'n = ', n
      write(*,*) ''
      do 10 i=1,n
        write(*,'(A,I3,A,F20.15,A,F20.15)')
     &    'i=', i, ' x=', x(i), ' w=', w(i)
10    continue

c     Compute integral of x^2
      sum = 0.0d0
      do 20 i=1,n
        sum = sum + w(i) * x(i)**2
20    continue

      exact = 1.0d0 / 3.0d0
      error = abs(sum - exact)

      write(*,*) ''
      write(*,*) 'Integration test: integral of x^2 from 0 to 1'
      write(*,'(A,F20.15)') 'Computed: ', sum
      write(*,'(A,F20.15)') 'Exact:    ', exact
      write(*,'(A,E15.8)')  'Error:    ', error

      end

c     Include the actual subroutine
      include 'gauss_func.f'
