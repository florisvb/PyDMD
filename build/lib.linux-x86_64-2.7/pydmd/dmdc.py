"""
Derived module from dmdbase.py for dmd with control.

Reference:
- Proctor, J.L., Brunton, S.L. and Kutz, J.N., 2016. Dynamic mode decomposition
with control. SIAM Journal on Applied Dynamical Systems, 15(1), pp.142-161.
"""
from .dmdbase import DMDBase
import numpy as np
import copy


def matrix_inv(X, max_sigma=1e-3):
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    Sigma_inv = Sigma**-1
    Sigma_inv[np.where(Sigma<max_sigma)[0]] = 0 # helps reduce instabilities
    return V.T.dot(np.diag(Sigma_inv)).dot(U.T)

class DMDc(DMDBase):
    """
    Dynamic Mode Decomposition with control.
    This version does not allow to manipulate the temporal window within the
    system is reconstructed.

    :param svd_rank: the rank for the truncation; If 0, the method computes the
        optimal rank and uses it for truncation; if positive interger, the
        method uses the argument for the truncation; if float between 0 and 1,
        the rank is the number of the biggest singular values that are needed
        to reach the 'energy' specified by `svd_rank`; if -1, the method does
        not compute truncation.
    :type svd_rank: int or float
    :param int tlsq_rank: rank truncation computing Total Least Square. Default
        is 0, that means no truncation.
    :param bool opt: flag to compute optimized DMD. Default is False.
    """

    def __init__(self, svd_rank=0, tlsq_rank=0, opt=False):
        self.svd_rank = svd_rank
        self.tlsq_rank = tlsq_rank
        self.opt = opt
        self.original_time = None

        self._eigs = None
        self._Atilde = None
        self._Btilde = None
        self._modes = None  # Phi
        self._b = None  # amplitudes
        self._snapshots = None
        self._snapshots_shape = None
        self._controlin = None
        self._controlin_shape = None

    def forward_backward_reconstruct(self, x0, forward_steps, backward_steps, controls=None, start=0, overlap=0):
        """

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """
        assert type(x0) == np.matrix 

        Atilde = np.matrix(self._Atilde)
        #Btilde = np.matrix(self._Btilde)
        U = np.matrix(self.U)
        B = self.B #U*Btilde*U.T.conj()
        

        UAUT = U*Atilde*U.T.conj()
        #invUAUT = matrix_inv(U.T.conj())*matrix_inv(Atilde)*matrix_inv(U) #    UAUT.I #matrix_inv(UAUT)
        invUAUT = matrix_inv(UAUT)

        # forward            
        x_reconstructed = copy.copy(x0)
        for i in range(1,forward_steps):
            if controls is None:
                u = np.zeros_like(x0)
            else:
                idx = start+i
                if idx < controls.shape[1] and idx > 0:
                    u = controls[:, idx]
                else:
                    u = np.zeros_like(x0)
            x = x_reconstructed[:,-1]
            x_f = UAUT*x + B*u
            x_reconstructed = np.hstack((x_reconstructed, x_f))

        x_reconstructed = x_reconstructed[:,overlap:] # first few points are bad??? and it's offset by 1

        # backward
        if 1:
            for i in range(0,backward_steps+overlap):
                if controls is None:
                    u = np.zeros_like(x0)
                else:
                    idx = start-i
                    if idx < controls.shape[1] and idx > 0:
                        u = controls[:, idx]
                    else:
                        u = np.zeros_like(x0)
                x = x_reconstructed[:,0]
                x_b = invUAUT*(x - B*u)
                x_reconstructed = np.hstack((x_b, x_reconstructed))


        return x_reconstructed


    def reconstructed_data(self, x0=None, u=None):
        """
        Get the reconstructed data.

        :return: the matrix that contains the reconstructed snapshots.
        :rtype: numpy.ndarray
        """

        if x0 is not None:
            x_reconstructed = copy.copy(x0)
            for i in range(1,u.shape[1]-1):
                _u_ = u[:,i-1]
                x = x_reconstructed[:,-1]

                x = (self.U.dot(self._Atilde)).dot(self.U.T.conj().dot(x)) + self.B.dot(_u_)

                x_reconstructed = np.hstack((x_reconstructed, np.reshape(x, [len(x), 1])))

            return x_reconstructed

        if x0 is None and u is not None:
            x_reconstructed = np.reshape(self._snapshots[:, 0], [self._snapshots.shape[0], 1])
            for i in range(1,u.shape[1]-1):
                _u_ = u[:,i-1]
                x = x_reconstructed[:,-1]

                x = (self.U.dot(self._Atilde)).dot(self.U.T.conj().dot(x)) + self.B.dot(_u_)

                x_reconstructed = np.hstack((x_reconstructed, np.reshape(x, [len(x), 1])))

            # now fix the first 2 points



            return x_reconstructed


        x_reconstructed = np.reshape(self._snapshots[:, 0], [self._snapshots.shape[0], 1])
        for i in range(1,self._snapshots.shape[1]-1):
            u = self._controlin[:,i-1]
            x = x_reconstructed[:,-1]

            x = (self.U.dot(self._Atilde)).dot(self.U.T.conj().dot(x)) + self.B.dot(u)

            x_reconstructed = np.hstack((x_reconstructed, np.reshape(x, [len(x), 1])))

        return x_reconstructed

        #return self.modes.dot(self.dynamics)

        if 0:
            try: # B unknown case
                return self.Ur.dot(self._Atilde).dot(
                    self.Ur.dot(self._snapshots[:, :-1])) + self.Ur.dot(
                        self._Btilde).dot(self._controlin)
            except AttributeError: # B known case
                return self._Atilde.dot(self._snapshots[:, :-1]) + self._Btilde.dot(
                    self._controlin)

    @property
    def btilde(self):
        """
        Get the reduced operator B.

        :return: the reduced operator B.
        :rtype: numpy.ndarray
        """
        return self._Btilde

    def _fit_B_known(self, X, Y, I, B):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is provided.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        :param numpy.ndarray I: the input control matrix.
        :param numpy.ndarray B: the matrib B.
        """

        X, Y = self._compute_tlsq(X, Y, self.tlsq_rank)

        U, s, V = self._compute_svd(X, self.svd_rank)
        self.U = U
        self.B = B

        #self._Atilde = (Y - B.dot(self._controlin)).dot(V).dot(
        #    np.diag(np.reciprocal(s))).dot(U.T.conj())

        self._Atilde = self._build_lowrank_op(U, s, V, Y - B.dot(self._controlin))

        self._eigs, self._modes = self._eig_from_lowrank_op(
            self._Atilde, (Y - B.dot(self._controlin)), U, s, V, True)

        self._b = self._compute_amplitudes(self._modes, self._snapshots,
                                           self._eigs, self.opt)

        

        self._Btilde = U.T.conj().dot(B)

        return self

    def _fit_B_unknown(self, X, Y, I):
        """
        Private method that performs the dynamic mode decomposition algorithm
        with control when the matrix `B` is not provided.

        :param numpy.ndarray X: the first matrix of original snapshots.
        :param numpy.ndarray Y: the second matrix of original snapshots.
        :param numpy.ndarray I: the input control matrix.
        """

        omega = np.vstack([X, self._controlin])

        Up, sp, Vp = self._compute_svd(omega, self.svd_rank)

        Up1 = Up[:self._snapshots.shape[0], :]
        Up2 = Up[self._snapshots.shape[0]:, :]
        # TODO: a second svd_rank?
        Ur, sr, Vr = self._compute_svd(Y, -1)

        self.Ur = Ur
        self.U = Ur

        self._Atilde = Ur.T.conj().dot(Y).dot(Vp).dot(
            np.diag(np.reciprocal(sp))).dot(Up1.T.conj()).dot(Ur)
        self._Btilde = Ur.T.conj().dot(Y).dot(Vp).dot(
            np.diag(np.reciprocal(sp))).dot(Up2.T.conj())

        self._eigs, modes = np.linalg.eig(self._Atilde)
        self._modes = Y.dot(Vp).dot(np.diag(np.reciprocal(sp))).dot(
            Up1.T.conj()).dot(Ur).dot(modes)

        self._b = self._compute_amplitudes(self._modes, X, self._eigs, self.opt)

    def fit(self, X, I, B=None):
        """
        Compute the Dynamic Modes Decomposition with control given the original
        snapshots and the control input data. The matrix `B` that controls how
        the control input influences the system evolution can be provided by
        the user; otherwise, it is computed by the algorithm.

        :param X: the input snapshots.
        :type X: numpy.ndarray or iterable
        :param I: the control input.
        :type I: numpy.ndarray or iterable
        :param numpy.ndarray B: 
        """
        self._snapshots, self._snapshots_shape = self._col_major_2darray(X)
        self._controlin, self._controlin_shape = self._col_major_2darray(I)

        n_samples = self._snapshots.shape[1]
        X = self._snapshots[:, :-1]
        Y = self._snapshots[:, 1:]

        if B is None:
            self._fit_B_unknown(X, Y, I)
        else:
            self._fit_B_known(X, Y, I, B)

        # Default timesteps
        n_samples = self._snapshots.shape[1]
        self.original_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}
        self.dmd_time = {'t0': 0, 'tend': n_samples - 1, 'dt': 1}