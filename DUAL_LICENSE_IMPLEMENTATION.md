# Dual Licensing Implementation Summary

**Date**: December 11, 2024
**Repository**: https://github.com/pageman/MILES-DiscoveryRL
**Compliance**: Apache 2.0 (MILES) + MIT (New Contributions)

---

## ✅ Implementation Complete

Your repository now properly implements **dual licensing** to comply with both the Apache 2.0 license from the MILES framework and MIT license for your new contributions.

---

## 📋 What Was Changed

### 1. License Files Added/Modified

✅ **LICENSE.apache** (New)
- Full text of Apache License 2.0
- Applies to MILES framework-derived concepts
- Required for Apache 2.0 compliance

✅ **LICENSE.mit** (Renamed from LICENSE)
- Full text of MIT License
- Copyright (c) 2024 Paul Pajo and Contributors
- Applies to all new drug discovery RL contributions

✅ **NOTICE** (New)
- Required by Apache 2.0 license
- Contains attribution to MILES framework
- Lists third-party dependencies
- Explains licensing structure

### 2. Source Code Updated

✅ **original/miles_concepts_drug_rl.py**
- Added Apache 2.0 license header
- Includes copyright notices for both MILES framework and your contributions
- References original MILES repository
- Complies with Apache 2.0 attribution requirements

Example header added:
```python
# Copyright 2024 Paul Pajo and Contributors
#
# Portions of this file are derived from concepts in the MILES framework:
# Copyright (c) 2024 MILES Framework Contributors
# https://github.com/radixark/miles
#
# Licensed under the Apache License, Version 2.0...
```

### 3. Documentation Updated

✅ **README.md**
- Added prominent "License Notice" section at the top
- Clearly explains dual licensing structure
- Lists which files are under which license
- Provides links to both license files and NOTICE
- Added "Citation & Licensing" section with:
  - BibTeX citation format
  - Attribution requirements for both licenses
  - Third-party dependency acknowledgments

---

## 🎯 Licensing Structure

### Files Under Apache 2.0 License

**Location**: `original/miles_concepts_drug_rl.py`

**Why**: This file contains concepts derived from or inspired by the MILES framework (Mixture of Experts architecture, distributed training concepts).

**Requirements When Using**:
- ✅ Include LICENSE.apache in distributions
- ✅ Include NOTICE file in distributions
- ✅ Retain all copyright and attribution notices
- ✅ State any modifications made
- ✅ Link to original MILES framework: https://github.com/radixark/miles

**Permissions**:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Patent grant included

### Files Under MIT License

**Location**: All other files including:
- `core/Drug_Optimization_RL_Enhanced.ipynb`
- `core/drug_rl_enhanced_analysis.py`
- `core/drug_rl_enhanced_notebook.py`
- `original/drug_rl_environment.py`
- `original/drug_rl_training.py`
- `original/drug_target_analysis.py`
- All `docs/` files
- All `examples/` files
- `README.md`, `FILE_MANIFEST.txt`, etc.

**Why**: These are original contributions specific to drug discovery RL, not derived from MILES.

**Requirements When Using**:
- ✅ Include LICENSE.mit in distributions
- ✅ Retain copyright notice: (c) 2024 Paul Pajo and Contributors

**Permissions**:
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ✅ Sublicensing allowed

---

## 🔍 License Compatibility

### Apache 2.0 + MIT = ✅ Compatible

Both licenses are:
- **Permissive** (not copyleft like GPL)
- **Commercial-friendly** (allow commercial use)
- **Compatible** with each other (can be used together)
- **OSI-approved** (recognized open source licenses)

### Key Differences

| Aspect | Apache 2.0 | MIT |
|--------|------------|-----|
| **Patent Grant** | ✅ Explicit patent grant | ❌ No explicit patent clause |
| **Attribution** | ✅ Requires NOTICE file | ✅ Requires copyright notice |
| **Trademark** | ✅ Explicitly not granted | Silent on trademarks |
| **Modifications** | Must be documented | No requirement |
| **Complexity** | More detailed (~300 lines) | Very simple (~20 lines) |

### Why This Matters

1. **Apache 2.0 provides stronger patent protection** - Important for MILES framework concepts
2. **MIT is simpler and more permissive** - Easier for users of your drug RL code
3. **Both allow commercial use** - No barriers to industry adoption
4. **Dual licensing respects both origins** - Legal and ethical compliance

---

## 📄 Required Notices

### When Distributing Your Code

Users must include:

1. **For Apache 2.0 portions**:
   ```
   LICENSE.apache (full text)
   NOTICE (attribution file)
   ```

2. **For MIT portions**:
   ```
   LICENSE.mit (full text with copyright)
   ```

3. **Best Practice**:
   - Include both license files
   - Include NOTICE file
   - Keep this README.md (explains structure)
   - Preserve all copyright headers in source files

### Citation in Academic Papers

```bibtex
@software{pajo2024miles_discoveryrl,
  author = {Pajo, Paul and Contributors},
  title = {MILES-DiscoveryRL: Drug Optimization with Reinforcement Learning},
  year = {2024},
  url = {https://github.com/pageman/MILES-DiscoveryRL},
  note = {Dual-licensed: Apache 2.0 (MILES concepts) and MIT (new contributions)}
}
```

Also cite:
- **MILES framework**: https://github.com/radixark/miles (Apache 2.0)
- **EvE Bio dataset**: https://huggingface.co/datasets/eve-bio/drug-target-activity
- **Discovery2 models**: https://huggingface.co/pageman (your repositories)

---

## ✅ Compliance Checklist

### Apache 2.0 Compliance
- [x] Apache 2.0 license text included (LICENSE.apache)
- [x] NOTICE file created with attribution
- [x] Copyright notices in MILES-derived files
- [x] Link to original MILES repository
- [x] Attribution requirements documented

### MIT License Compliance
- [x] MIT license text included (LICENSE.mit)
- [x] Copyright notice: (c) 2024 Paul Pajo and Contributors
- [x] License applies to original contributions

### Documentation
- [x] README explains licensing structure
- [x] Prominent notice at top of README
- [x] Citation requirements documented
- [x] Attribution requirements clear
- [x] Links to license files provided

### Legal Protection
- [x] Both licenses are OSI-approved
- [x] Commercial use permitted by both
- [x] Patent grant from Apache 2.0
- [x] No license conflicts
- [x] Proper attribution chain maintained

---

## 🎓 For Users of This Repository

### If You're Just Using the Code

**Simple approach**:
1. Clone the repository
2. Keep both LICENSE.apache and LICENSE.mit files
3. Keep the NOTICE file
4. Follow the usage instructions in README.md

**You're covered** - both licenses are very permissive!

### If You're Modifying the Code

**For Apache 2.0 files** (`original/miles_concepts_drug_rl.py`):
- Document your changes (comments or changelog)
- Keep the Apache 2.0 header
- Include LICENSE.apache and NOTICE in your distribution

**For MIT files** (everything else):
- Keep the MIT license header/notice
- Include LICENSE.mit in your distribution
- That's it! MIT is very simple.

### If You're Distributing the Code

**Include these files**:
- LICENSE.apache
- LICENSE.mit
- NOTICE
- README.md (explains structure)
- All source files with their headers intact

### If You're Publishing Research

**Cite**:
- This repository (BibTeX above)
- MILES framework
- EvE Bio dataset
- Discovery2 models

**Acknowledge**:
- Dual licensing (Apache 2.0 + MIT)
- Original MILES framework authors

---

## 🚀 Benefits of Dual Licensing

### For You (Repository Owner)
✅ Legal compliance with MILES Apache 2.0
✅ MIT licensing makes your code more accessible
✅ Clear separation of derived vs. original work
✅ Protection from patent claims (Apache 2.0)
✅ Flexibility for future contributions

### For Users
✅ Can use for commercial purposes
✅ Can modify and redistribute
✅ Clear licensing structure (no ambiguity)
✅ Patent protection (Apache 2.0)
✅ Simple compliance requirements

### For the Community
✅ Respects original MILES authors
✅ Encourages open source contribution
✅ Transparent attribution
✅ Industry-friendly licensing
✅ Academic publication compatible

---

## 📞 Questions About Licensing?

### Common Questions

**Q: Can I use this commercially?**
A: Yes! Both Apache 2.0 and MIT allow commercial use.

**Q: Do I need to open source my modifications?**
A: No, neither license requires that (they're not copyleft).

**Q: What if I only use the MIT-licensed parts?**
A: Then you only need to include LICENSE.mit and the copyright notice.

**Q: Can I relicense my modifications?**
A: For MIT parts, yes (within license terms). For Apache 2.0 parts, you must keep Apache 2.0.

**Q: Do I need a lawyer?**
A: Both are standard OSI-approved licenses, widely used and well-understood. For complex situations, consult legal counsel.

### Resources

- **Apache 2.0 License**: https://www.apache.org/licenses/LICENSE-2.0
- **MIT License**: https://opensource.org/licenses/MIT
- **OSI License Guide**: https://opensource.org/licenses
- **GitHub Licensing Guide**: https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/licensing-a-repository

### Contact

For licensing questions specific to this repository:
- Open an issue: https://github.com/pageman/MILES-DiscoveryRL/issues
- Check the README: https://github.com/pageman/MILES-DiscoveryRL#license-notice-important

---

## 🎉 Summary

Your MILES-DiscoveryRL repository now has:

✅ **Proper dual licensing** (Apache 2.0 + MIT)
✅ **Full compliance** with MILES framework license
✅ **Clear documentation** of licensing structure
✅ **Required notices** (LICENSE files + NOTICE)
✅ **Proper attribution** to MILES framework
✅ **Commercial-friendly** licensing
✅ **Academic-friendly** citation requirements

**Status**: ✅ Legally compliant and ready for public use!

---

**Last Updated**: December 11, 2024
**Repository**: https://github.com/pageman/MILES-DiscoveryRL
**Licenses**: Apache 2.0 (LICENSE.apache) + MIT (LICENSE.mit)
**Attribution**: See NOTICE file
