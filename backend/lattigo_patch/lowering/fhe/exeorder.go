package main

import (
	"fmt"
)

type PQItem struct {
	Index  int
	Value  int
	Value2 int // secondary value for tie-breaking if needed
}

func (item PQItem) Less(other PQItem) bool {
	if item.Value == other.Value {
		return item.Value2 > other.Value2
	}
	return item.Value > other.Value
}

type PriorityQueue struct {
	items []PQItem
}

func NewPriorityQueue() *PriorityQueue {
	return &PriorityQueue{
		items: make([]PQItem, 0),
	}
}

func (pq *PriorityQueue) Len() int {
	return len(pq.items)
}

func (pq *PriorityQueue) Insert(index, value, value2 int) {
	item := PQItem{Index: index, Value: value, Value2: value2}
	pq.items = append(pq.items, item)
	pq.upHeap(pq.Len() - 1)
}

func (pq *PriorityQueue) Min() (int, int, bool) {
	if pq.Len() == 0 {
		return 0, 0, false
	}
	item := pq.items[0]
	return item.Index, item.Value, true
}

func (pq *PriorityQueue) DeleteMin() (int, int, bool) {
	if pq.Len() == 0 {
		return 0, 0, false
	}
	minItem := pq.items[0]
	last := pq.Len() - 1
	pq.items[0] = pq.items[last]
	pq.items = pq.items[:last]
	pq.downHeap(0)
	return minItem.Index, minItem.Value, true
}

func (pq *PriorityQueue) upHeap(i int) {
	for {
		parent := (i - 1) / 2
		if i == 0 || pq.items[parent].Less(pq.items[i]) {
			break
		}
		pq.items[parent], pq.items[i] = pq.items[i], pq.items[parent]
		i = parent
	}
}

func (pq *PriorityQueue) downHeap(i int) {
	n := pq.Len()
	for {
		left := 2*i + 1
		right := 2*i + 2
		smallest := i
		if left < n && pq.items[left].Less(pq.items[smallest]) {
			smallest = left
		}
		if right < n && pq.items[right].Less(pq.items[smallest]) {
			smallest = right
		}
		if smallest == i {
			break
		}
		pq.items[i], pq.items[smallest] = pq.items[smallest], pq.items[i]
		i = smallest
	}
}
func (lattigo *LattigoFHE) getOptimalRunOrder(numOps int) ([]int, []int, []int) {

	// fmt.Println("Getting optimal run order for", numOps, "operations")
	depth := make([]int, numOps)
	back_depth := make([]int, numOps)
	outputs := make([][]int, numOps)
	for lineNum := range numOps {
		term := lattigo.terms[lineNum]
		prev_depth := -1
		for _, child := range term.Children {
			if child < 0 {
				prev_depth = max(prev_depth, 0)
			} else {
				prev_depth = max(prev_depth, depth[child])
				outputs[child] = append(outputs[child], lineNum)
			}
		}
		depth[lineNum] = prev_depth + 1
	}
	// for lineNum := range numOps {
	// 	if depth[lineNum] == 0 {
	// 		depth[lineNum] = numOps
	// 		for _, output := range outputs[lineNum] {
	// 			depth[lineNum] = min(depth[lineNum], depth[output]-1)
	// 		}
	// 	}
	// }
	for lineNum := range numOps {
		if len(lattigo.terms[lineNum].Children) == 0 {
			back_depth[lineNum] = 0
		}
	}
	for lineNum := numOps - 1; lineNum >= 0; lineNum-- {
		term := lattigo.terms[lineNum]
		for _, child := range term.Children {
			if child >= 0 {
				back_depth[child] = max(back_depth[child], back_depth[lineNum]+1)
			}
		}
	}
	pq := NewPriorityQueue()
	remaining_inputs := make([]int, numOps)
	order_list := make([]int, 0)
	for lineNum := range numOps {
		term := lattigo.terms[lineNum]
		hasPositiveChild := false
		for _, child := range term.Children {
			if child >= 0 {
				hasPositiveChild = true
				remaining_inputs[lineNum]++
			}
		}
		if !hasPositiveChild {
			pq.Insert(lineNum, back_depth[lineNum], depth[lineNum])
		}
	}
	for pq.Len() > 0 {
		lineNum, _, _ := pq.DeleteMin()
		order_list = append(order_list, lineNum)
		for _, out := range outputs[lineNum] {
			remaining_inputs[out]--
			if remaining_inputs[out] == 0 {
				pq.Insert(out, back_depth[out], depth[out])
			}
		}
	}
	return order_list, depth, back_depth
}

func (lattigo *LattigoFHE) getMaxWorkingPoolSize(numOps int, order_list []int) (int, error) {
	// First check order_list is a valid topological order
	isExist := make(map[int]bool)
	for i := range numOps {
		if _, ok := isExist[order_list[i]]; ok {
			return 0, fmt.Errorf("line %d appears multiple times in execution order", order_list[i])
		}
		isExist[order_list[i]] = true
	}
	for i := range numOps {
		if _, ok := isExist[i]; !ok {
			return 0, fmt.Errorf("line %d missing in execution order", i)
		}
	}

	working_pool_size := 0
	max_working_pool_size := 0
	outputCounts := make(map[int]int)
	appeared := make(map[int]bool)
	for lineNum := range numOps {
		term := lattigo.terms[lineNum]
		for _, child := range term.Children {
			if child >= 0 {
				outputCounts[child]++
			} else {
				// working_pool_size = max(working_pool_size, -child)
				appeared[child] = true
			}
		}
	}
	for i := range numOps {
		lineNum := order_list[i]
		term := lattigo.terms[lineNum]
		appeared[lineNum] = true
		for _, child := range term.Children {
			if !appeared[child] {
				return 0, fmt.Errorf("child %d of term %d has not appeared before its parent", child, lineNum)
			}
		}
		if len(term.Children) > 0 {
			working_pool_size++
		}
		max_working_pool_size = max(max_working_pool_size, working_pool_size)
		for _, child := range term.Children {
			if child >= 0 {
				outputCounts[child]--
				if outputCounts[child] == 0 {
					if len(lattigo.terms[child].Children) > 0 {
						working_pool_size--
					}
					// working_pool_size--
				}
			}
		}
	}
	return max_working_pool_size, nil

}
