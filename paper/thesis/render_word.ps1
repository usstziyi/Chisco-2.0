$ErrorActionPreference = 'Stop'
$thesisPath = Join-Path $PSScriptRoot '毕业论文完整初稿.docx'
$qaPath = Join-Path $PSScriptRoot 'qa'
New-Item -ItemType Directory -Path $qaPath -Force | Out-Null
$pdfPath = Join-Path $qaPath 'thesis.pdf'
$word = $null
$document = $null
try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $document = $word.Documents.Open($thesisPath, $false, $false)
    $document.Fields.Update() | Out-Null
    foreach ($toc in $document.TablesOfContents) {
        $toc.Update()
        $toc.Range.Font.Size = 11
        $toc.Range.ParagraphFormat.LineSpacingRule = 4
        $toc.Range.ParagraphFormat.LineSpacing = 16
        $toc.Range.ParagraphFormat.SpaceBefore = 0
        $toc.Range.ParagraphFormat.SpaceAfter = 3
    }
    $document.Repaginate()
    foreach ($toc in $document.TablesOfContents) { $toc.UpdatePageNumbers() }
    $document.Save()
    $document.ExportAsFixedFormat($pdfPath, 17)
    $pageCount = $document.ComputeStatistics(2)
    Write-Output "Pages=$pageCount PDF=$pdfPath TOC=$($document.TablesOfContents.Count)"
} finally {
    if ($null -ne $document) { $document.Close(0) }
    if ($null -ne $word) { $word.Quit() }
}
