# Library coverage

*Part of the [AuToGraFS documentation](../README.md#documentation).*

2593 of the 2686 shipped topologies (**96.5 %**) have at least one compatible
SBU for every slot type; `scripts/sbu_coverage.py` reproduces the number.
Compatibility is a deliberately *permissive* geometric sieve — an SBU is
listed when its connection-vector shape matches the slot's to within a
directional RMSD of 0.35 (square vs tetrahedral scores ~0.6). The sieve says
what is worth trying; structure *quality* is enforced where it belongs, at
build time, by `max_rmsd` and `min_distance`, and distorted-but-valid output
can be cleaned up with `Framework.relax()`.

<details>
<summary>The 93 topologies not currently buildable (click to expand)</summary>

- **50 nets whose vertex figure no current SBU matches** (best match above
  0.35): awd, dnb, dnd, dno, dns, eca, eck, eee, hch, hci, hcz, hcz-a, hxg-d,
  jak, jmt, ken, mte, ncb, ncd, ncg, nci, ncj, ncl, ncm, nia-d, ntu, sde, sep,
  skg, srr, swn, ton, tsn, ttr, ttt, utx, uty, vcx, vna, vne, wal, wyt, xay,
  xbc, xbn, xbp, xbr, xbs, xbz, zim.
- **43 nets with a vertex connectivity no library covers at all** (mostly
  augmented `-x` and dual `-d` variants of nets whose parent form *is*
  covered):
  - 11-c: ela, elb, elc, eld, ele, elf, lwa, lwa-d, mjt, nin, svi-x
  - 13-c: amn, nas
  - 14-c: bcu-x, bem, bet, gpu-x, jkz, kcz, keb, nin, nts-d, reo-d, tcc-x,
    tcf-x, tcg-x, wzz, zra
  - 15-c: cal, cla-d, zra
  - 16-c: amn, dia-x, mgc-x, mgz-x, nas, uro, urq, urs
  - 17-c: odf-d
  - 18-c: ast-d, gea, gez, nts-d, she-d, ytw
  - 20-c: alb-x, ccu

</details>

If you care about one of these, a user-supplied building block with the right
connectivity makes it available — see
[Extending the libraries](extending.md#custom-building-blocks-sbus).
