    SUBROUTINE CONFIG(NY,EPSILON,SGM,PMLY,DY,W_CU,W_PY,JB1,JB2,J1,J2,SIGMA1,SIGMA2,EPS_INF)
	
	IMPLICIT NONE
	
	INTEGER NY, PMLY, FLAG_FCR
	REAL*8 EPSILON(NY+1), SGM(NY+1)
	INTEGER J, CENTER_Y, JB1, JB2, J1, J2
	REAL*8 DY
	REAL*8 W_CU, W_PY
	REAL*8 SIGMA1, SIGMA2, EPS_INF

	CENTER_Y = NY/2

	J1 = CENTER_Y+1-CEILING(W_PY/DY/2.0D0)
    JB1 = CENTER_Y+1+CEILING(W_PY/DY/2.0D0)
		
	WRITE(*,*) 'TOP Py BOUNDARY', J1
    WRITE(*,*) 'BOTTOM Py BOUNDARY', JB1-1

!	PERMALLOY, FREE LAYER
	DO J=J1, JB1-1

		EPSILON(J) = EPS_INF
		SGM(J) = SIGMA1
                        
    END DO
    
	OPEN (UNIT=34,FILE='SGM.OUT',STATUS='UNKNOWN')
        DO J=1, NY+1
                WRITE(34,"(1X, E18.8E3)")  SGM(J)

        ENDDO
        CLOSE(34)

	WRITE(*,*) 'CONFIG OK!'
	
	END
