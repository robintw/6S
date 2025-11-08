      program test_possol
c     Test program for POSSOL (solar position calculation)
      integer month, jday
      real tu, xlon, xlat, asol, phi0

c     Test case: June 21, 2000 (summer solstice)
c     Location: Greenwich, UK (0°E, 51.5°N)
c     Time: 12:00 UTC (solar noon approximately)
      month = 6
      jday = 21
      tu = 12.0
      xlon = 0.0
      xlat = 51.5

      call possol(month, jday, tu, xlon, xlat, asol, phi0)

      write(*,*) 'Solar position calculation test'
      write(*,*) ''
      write(*,'(A,I2,A,I2)') 'Date: ', month, '/', jday
      write(*,'(A,F10.4)') 'Time (UTC): ', tu
      write(*,'(A,F10.4,A)') 'Longitude: ', xlon, ' degrees'
      write(*,'(A,F10.4,A)') 'Latitude: ', xlat, ' degrees'
      write(*,*) ''
      write(*,'(A,F10.4,A)') 'Solar zenith angle: ', asol, ' degrees'
      write(*,'(A,F10.4,A)') 'Solar azimuth angle: ', phi0, ' degrees'
      write(*,*) ''

c     Test case 2: Different location and time
      month = 12
      jday = 21
      tu = 14.0
      xlon = -122.4  ! San Francisco
      xlat = 37.8

      call possol(month, jday, tu, xlon, xlat, asol, phi0)

      write(*,*) 'Test case 2:'
      write(*,'(A,I2,A,I2)') 'Date: ', month, '/', jday
      write(*,'(A,F10.4)') 'Time (UTC): ', tu
      write(*,'(A,F10.4,A)') 'Longitude: ', xlon, ' degrees'
      write(*,'(A,F10.4,A)') 'Latitude: ', xlat, ' degrees'
      write(*,*) ''
      write(*,'(A,F10.4,A)') 'Solar zenith angle: ', asol, ' degrees'
      write(*,'(A,F10.4,A)') 'Solar azimuth angle: ', phi0, ' degrees'

      end

c     Include the actual subroutines
      include 'possol_funcs.f'
