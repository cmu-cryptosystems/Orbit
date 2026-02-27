from ..tdag.tdag import Tdag
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

try:
    import pydot
    PYDOT_AVAILABLE = True
except ImportError:
    PYDOT_AVAILABLE = False


def comp_color(color:str):
    rgb = mcolors.to_rgb(color)
    comp_rgb = tuple(1.0 - c for c in rgb)
    comp_hex = mcolors.to_hex(comp_rgb)
    return comp_hex
    

def visualize(tdag: Tdag, filename: str):
    if not PYDOT_AVAILABLE:
        print("Error: pydot is required for visualization. Install with: pip install pydot")
        return False
    
    graph = pydot.Dot(graph_type='digraph', rankdir='TB')
    op_colors = {
        'constant': 'white',
        'input': 'black',
        'output': 'lightgrey',
        'dummy': 'black',
        'add': 'lightblue',
        'negate': 'lightblue',
        'mul': 'blue',
        'rotate': 'purple',
        'rescale': 'red',
        'upscale': 'lightgreen',
        'modswitch': 'darkgreen',
        'bootstrap': 'yellow'
    }
    
    for v in tdag.nodes:
        vattr = tdag.nodes[v]
        color = op_colors.get(vattr['op'], 'white')
        label_parts = [f"{v}: {vattr['weight']}x {vattr['op']}"]
        label_parts.append(f"({vattr['scale']} * {vattr['level']})")
        label_parts.append(f"cmt: {vattr['comment']}")
        if vattr['op'] == 'mul' and vattr['op_descr']['double']:
            color = 'darkblue'
        
        label = "\\n".join(label_parts)

        node = pydot.Node(v, label=label, style='filled', fillcolor=color, fontcolor=comp_color(color), shape='box')
        graph.add_node(node)
    
    for u, v, eattr in tdag.edges(data=True):
        edge_label = str(eattr.get('weight', ''))
        graph.add_edge(pydot.Edge(u, v, label=edge_label))
    
    print("Writing visualization to", filename)
    graph.write_svg(filename, prog='dot')
    print("Visualization saved.")

def visualize_with_region(tdag: Tdag, filename: str):
    if not PYDOT_AVAILABLE:
        print("Error: pydot is required for visualization. Install with: pip install pydot")
        return False
    
    # Use rankdir='TB' for a top-to-bottom layout
    graph = pydot.Dot(graph_type='digraph', rankdir='TB')
    
    op_colors = {
        'constant': 'white',
        'input': 'black',
        'output': 'lightgrey',
        'dummy': 'black',
        'add': 'lightblue',
        'negate': 'lightblue',
        'mul': 'blue',
        'rotate': 'purple',
        'rescale': 'red',
        'upscale': 'lightgreen',
        'modswitch': 'darkgreen',
        'bootstrap': 'yellow'
    }
    regions = dict()
    for v in tdag.nodes:
        vattr = tdag.nodes[v]
        region_id = int(vattr.get('comment', 0))
        if region_id not in regions:
            regions[region_id] = []
        regions[region_id].append(v)

    sorted_regions = sorted(regions.keys())
    for u,v in tdag.edges:
        ru = int(tdag.nodes[u].get('comment', 0))
        rv = int(tdag.nodes[v].get('comment', 0))
        if ru > rv:
            raise ValueError(f"Edge from higher region {ru} to lower region {rv}: {u} -> {v}")

    for region_id in sorted_regions:
        subgraph = pydot.Subgraph(f'cluster_{region_id}', 
                                  label=f'Region {region_id}', 
                                  style='filled',
                                  color='lightgrey')
        
        subgraph.set('rank', 'same')

        for v in regions[region_id]:
            vattr = tdag.nodes[v]
            color = op_colors.get(vattr['op'], 'white')
            label_parts = [f"{v}: {vattr['weight']}x {vattr['op']}"]
            label_parts.append(f"({vattr['scale']} * {vattr['level']})")
            label_parts.append(f"cmt: {vattr['comment']}") 
            if vattr['op'] == 'mul' and vattr['op_descr']['double']:
                color = 'darkblue'
            
            label = "\\n".join(label_parts)

            node = pydot.Node(v, label=label, style='filled', fillcolor=color, fontcolor=comp_color(color), shape='box')
            
            subgraph.add_node(node)
        
        graph.add_subgraph(subgraph)
        
    for u, v, eattr in tdag.edges(data=True):
        edge_label = str(eattr.get('weight', ''))
        ru = int(tdag.nodes[u].get('comment', 0))
        rv = int(tdag.nodes[v].get('comment', 0))
        if ru == rv:
            graph.add_edge(pydot.Edge(u, v, label=edge_label, constraint='false'))
        else:
            col = 'blue' if ru == rv - 1 else 'red'
            graph.add_edge(pydot.Edge(u, v, label=edge_label, color=col))
    # dot_string = graph.to_string()
    # debug_dot_filename = filename.replace('.svg', '.dot')
    # # print(f"Saving DOT source for debugging to: {debug_dot_filename}")
    # # with open(debug_dot_filename, 'w') as f:
    # #     f.write(dot_string)
    
    print("Writing visualization to", filename)
    graph.write_svg(filename, prog='dot')
    print("Visualization saved.")
