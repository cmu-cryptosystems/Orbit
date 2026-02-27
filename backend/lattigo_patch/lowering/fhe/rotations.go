package main

import (
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

func (lattigo *LattigoFHE) decomposeRotation(rotation int) []int {
	if rotation == 0 {
		return []int{}
	}

	var decomposition []int

	sign := rotation < 0
	value := rotation
	if value < 0 {
		value = -value
	}

	for i := 0; value > 0; i++ {
		zi := 0
		if value&1 == 1 {
			zi = 2 - (value & 3)
		}
		value = (value - zi) >> 1

		if zi != 0 {
			term := zi * (1 << i)
			if sign {
				term = -term
			}
			if term%lattigo.n != 0 {
				decomposition = append(decomposition, term)
			}
		}
	}

	return decomposition
}

func (lattigo *LattigoFHE) notdoHoisted(childLineNum int) {
	baseCt := lattigo.env[childLineNum]

	for offset := range lattigo.hoistedRots[childLineNum] {
		decomposition := lattigo.decomposeRotation(offset)
		ct := baseCt
		for _, rot := range decomposition {
			ct, _ = lattigo.eval.RotateNew(ct, rot)
		}
		lattigo.hoistedRots[childLineNum][offset] = ct
	}
}

func (lattigo *LattigoFHE) doHoisted(childLineNum int) {
	baseCt := lattigo.env[childLineNum]

	// Get all required offsets
	offsets := make(map[int]bool)
	for offset := range lattigo.hoistedRots[childLineNum] {
		offsets[offset] = true
	}

	// Decompose all offsets into power-of-2 steps
	decompositions := make(map[int][]int)
	for offset, _ := range offsets {
		decompositions[offset] = lattigo.decomposeRotation(offset)
	}

	// Process decompositions index by index (left to right)
	results := lattigo.recurseHoisted(offsets, 0, decompositions, baseCt)

	// Store final results in hoistedRots
	for offset, finalCt := range results {
		lattigo.hoistedRots[childLineNum][offset] = finalCt
	}
}

// recurseHoisted recursively processes decompositions and builds rotation paths
func (lattigo *LattigoFHE) recurseHoisted(offsets map[int]bool, index int, decompositions map[int][]int, pathCiphertext *rlwe.Ciphertext) map[int]*rlwe.Ciphertext {
	valueGroups := make(map[int]map[int]bool)
	endedPaths := make(map[int]*rlwe.Ciphertext)
	toRemove := make([]int, 0)

	for offset := range offsets {
		if index < len(decompositions[offset]) {
			value := decompositions[offset][index]
			if valueGroups[value] == nil {
				valueGroups[value] = make(map[int]bool)
			}
			valueGroups[value][offset] = true
		} else {
			endedPaths[offset] = pathCiphertext
			toRemove = append(toRemove, offset)
		}
	}

	for _, offset := range toRemove {
		delete(offsets, offset)
		delete(decompositions, offset)
	}

	if len(valueGroups) == 1 {
		pathCiphertext = lattigo.evalRot(pathCiphertext, getKeys(valueGroups)[0])
		return mergeMaps(endedPaths, lattigo.recurseHoisted(offsets, index+1, decompositions, pathCiphertext))
	} else {
		rots, _ := lattigo.eval.RotateHoistedNew(pathCiphertext, getKeys(valueGroups))
		for rot, ct := range rots {
			rotOffsets := valueGroups[rot]
			recursiveResult := lattigo.recurseHoisted(rotOffsets, index+1, decompositions, ct)
			endedPaths = mergeMaps(endedPaths, recursiveResult)
		}
		return endedPaths
	}
}
