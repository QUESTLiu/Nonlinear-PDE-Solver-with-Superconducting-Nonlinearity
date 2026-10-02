2026-10-02 revision evidence

joint_phase_initial.json records the first 520-second execution. It is an
archived execution record, superseded for scientific selection by
double_sine_gordon_data/joint_phase_check.json.

Independent review identified two defects in the initial driver: qubit rotations
at word boundaries were merged only about Z instead of as complete SU(2)
matrices, and the hybrid boundary-population check omitted the point after native
propagation and before the first onsite word. The final driver fixes both.
It also tests the analytic alternating completion sign of arXiv:2609.40078v1,
Sec. III.4. All final data used by the manuscript are in joint_phase_check.json.

The initial record keeps its original numerical values to document the
executed calculation. Its primitive counts and completeness claim are not the
final validated result. It is not an input to figure generation or circuit
selection.
