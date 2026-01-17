package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
)

func calc_rms(values []float64) float64 {
	var sumSquares float64
	for _, v := range values {
		sumSquares += v * v
	}
	meanSquares := sumSquares / float64(len(values))
	return math.Sqrt(meanSquares)
}

func (lattigo *LattigoFHE) loadConstants(filename string) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("failed to open file %s: %v", filename, err)
	}
	defer file.Close()

	var length int64
	err = binary.Read(file, binary.LittleEndian, &length)
	if err != nil {
		return fmt.Errorf("failed to read length: %v", err)
	}

	const_rms := make([]float64, length)
	const_origin_len := make([]int, length)

	for i := int64(0); i < length; i++ {
		var veclen int64
		err = binary.Read(file, binary.LittleEndian, &veclen)
		if err != nil {
			return fmt.Errorf("failed to read vector length at index %d: %v", i, err)
		}
		var values []float64
		raw_values := make([]float64, veclen)
		err = binary.Read(file, binary.LittleEndian, &raw_values)
		if err != nil {
			return fmt.Errorf("failed to read vector values at index %d: %v", i, err)
		}
		const_rms[i] = calc_rms(raw_values)
		const_origin_len[i] = int(veclen)
		for j := int64(0); j < int64(lattigo.n); j++ {
			values = append(values, raw_values[j%veclen])
		}

		// if veclen > int64(lattigo.n) {
		// 	// Read only the first n values
		// 	values = make([]float64, lattigo.n)
		// 	err = binary.Read(file, binary.LittleEndian, &values)
		// 	if err != nil {
		// 		return fmt.Errorf("failed to read vector values at index %d: %v", i, err)
		// 	}
		// 	// Skip the remaining values by reading them into a temporary slice
		// 	remainingValues := make([]float64, veclen-int64(lattigo.n))
		// 	err = binary.Read(file, binary.LittleEndian, &remainingValues)
		// 	if err != nil {
		// 		return fmt.Errorf("failed to skip remaining vector values at index %d: %v", i, err)
		// 	}
		// } else {
		// 	// Read all values
		// 	values = make([]float64, veclen)
		// 	err = binary.Read(file, binary.LittleEndian, &values)
		// 	if err != nil {
		// 		return fmt.Errorf("failed to read vector values at index %d: %v", i, err)
		// 	}
		// }
		lattigo.constants[int(i)] = values
	}

	// const_rms_file, err := os.Create("const_rms.txt")
	// if err != nil {
	// 	return fmt.Errorf("failed to create const_rms.txt: %v", err)
	// }
	// defer const_rms_file.Close()

	// for i, rms := range const_rms {
	// 	_, err := fmt.Fprintf(const_rms_file, "Constant %d: len=%d RMS = %f\n", i, const_origin_len[i], rms)
	// 	if err != nil {
	// 		return fmt.Errorf("failed to write to const_rms.txt: %v", err)
	// 	}
	// }

	return nil
}
