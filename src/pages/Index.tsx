
import React, { useState } from 'react';
import JupyterHeader from '@/components/JupyterHeader';
import JupyterMenu from '@/components/JupyterMenu';
import JupyterToolbar from '@/components/JupyterToolbar';
import CodeCell from '@/components/CodeCell';

const Index = () => {
  const titleCode = `<h1>Interactive Python Visualization Libraries</h1>`;

  const cell1Code = `import numpy as np
import plotly.offline as py
import plotly.figure_factory as ff
from bokeh.models import HoverTool, WheelZoomTool
from bokeh.plotting import figure
from bokeh.io import show, output_notebook
output_notebook()`;

  const plotlyTitle = `<h2>Plotly</h2>`;

  const cell2Code = `py.init_notebook_mode()
t = np.linspace(-1, 1.2, 2000)
x = (t**3) + (0.3 * np.random.randn(2000))
y = (t**6) + (0.3 * np.random.randn(2000))

colorscale = ['#7A4579', '#D56073', 'rgb(236,158,105)', (1, 1, 0.2), (0.98,0.98,0.98)]

fig = ff.create_2d_density(
    x, y, colorscale=colorscale,
    hist_color='rgb(255, 237, 222)', point_size=3
)

py.iplot(fig, filename='histogram_subplots')`;

  const bokehTitle = `<h2>Bokeh</h2>`;

  const cell3Code = `n = 500
x = 2 + 2*np.random.standard_normal(n)
y = 2 + 2*np.random.standard_normal(n)

p = figure(title="Hexbin for 500 points", match_aspect=True,
         tools="wheel_zoom,pan,reset", background_fill_color='#440154')
p.grid.visible = False

r, bins = p.hexbin(x, y, size=0.5, hover_color="pink", hover_alpha=0.8)`;

  // Track which cell is active (for keyboard focus purposes)
  const [activeCellIndex, setActiveCellIndex] = useState(1);

  return (
    <div className="min-h-screen flex flex-col bg-white">
      <JupyterHeader />
      <JupyterMenu />
      <JupyterToolbar />
      
      <div className="flex-1 px-4 py-2 overflow-y-auto">
        <CodeCell cellNumber={0} code={titleCode} cellType="markdown" />
        
        <CodeCell 
          cellNumber={1} 
          code={cell1Code} 
          cellType="code" 
          isActive={activeCellIndex === 1} 
        />
        
        <CodeCell cellNumber={0} code={plotlyTitle} cellType="markdown" />
        
        <CodeCell 
          cellNumber={2} 
          code={cell2Code} 
          cellType="code" 
          isActive={activeCellIndex === 2} 
        />
        
        <CodeCell cellNumber={0} code={bokehTitle} cellType="markdown" />
        
        <CodeCell 
          cellNumber={3} 
          code={cell3Code} 
          cellType="code" 
          isActive={activeCellIndex === 3} 
        />
      </div>
    </div>
  );
};

export default Index;
