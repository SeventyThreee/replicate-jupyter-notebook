
import React, { useState } from 'react';

interface CodeCellProps {
  cellNumber: number;
  code: string;
  cellType: 'code' | 'markdown';
  isActive?: boolean;
}

const formatCode = (code: string) => {
  return code.split('\n').map((line, i) => {
    // Format Python syntax
    let formattedLine = line;
    
    // Format keywords
    formattedLine = formattedLine.replace(
      /(import|from|as|def|class|return|if|else|for|while|in|and|or|not|try|except|finally|with|lambda|None|True|False)(\s|$|:)/g,
      '<span class="code-keyword">$1</span>$2'
    );
    
    // Format imports specifically
    formattedLine = formattedLine.replace(
      /^(\s*)(import|from)\b/g,
      '$1<span class="code-import">$2</span>'
    );
    
    // Format the 'as' in import statements
    formattedLine = formattedLine.replace(
      /\b(as)\b/g,
      '<span class="code-as">$1</span>'
    );
    
    // Format strings
    formattedLine = formattedLine.replace(
      /(['"])(.*?)\1/g,
      '<span class="code-string">$1$2$1</span>'
    );
    
    // Format numbers
    formattedLine = formattedLine.replace(
      /\b(\d+\.?\d*)\b/g,
      '<span class="code-number">$1</span>'
    );
    
    return (
      <div key={i} className="flex">
        <div dangerouslySetInnerHTML={{ __html: formattedLine }} />
      </div>
    );
  });
};

const CodeCell: React.FC<CodeCellProps> = ({ cellNumber, code: initialCode, cellType, isActive = false }) => {
  const [code, setCode] = useState(initialCode);
  const [isEditing, setIsEditing] = useState(false);

  const handleDoubleClick = () => {
    if (cellType === 'code') {
      setIsEditing(true);
    }
  };

  const handleCodeChange = (e: React.ChangeEvent<HTMLTextAreaElement>) => {
    setCode(e.target.value);
  };

  const handleBlur = () => {
    setIsEditing(false);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // If Shift+Enter is pressed, exit editing mode
    if (e.key === 'Enter' && e.shiftKey) {
      setIsEditing(false);
    }
  };

  return (
    <div className={`jupyter-cell ${isActive ? 'jupyter-cell-active' : ''}`}>
      <div className="flex">
        <div className="w-14 bg-gray-100 text-gray-500 p-2 text-right font-mono text-sm">
          {cellType === 'code' ? `In [${cellNumber}]:` : ''}
        </div>
        <div 
          className="flex-1 p-2 font-mono text-sm whitespace-pre overflow-x-auto"
          onDoubleClick={handleDoubleClick}
        >
          {cellType === 'markdown' ? (
            <div className="prose max-w-none" dangerouslySetInnerHTML={{ __html: code }} />
          ) : isEditing ? (
            <textarea
              value={code}
              onChange={handleCodeChange}
              onBlur={handleBlur}
              onKeyDown={handleKeyDown}
              className="w-full h-full min-h-[100px] bg-transparent outline-none font-mono text-sm resize-none"
              autoFocus
            />
          ) : (
            <div>{formatCode(code)}</div>
          )}
        </div>
      </div>
    </div>
  );
};

export default CodeCell;
