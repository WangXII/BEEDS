<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:output method="text" encoding="UTF-8"/>

<xsl:template match="PubmedArticle">
        <xsl:value-of select="MedlineCitation/PMID"/>,<xsl:value-of select="MedlineCitation/Article/Journal/JournalIssue/PubDate/Year"/>,<xsl:value-of select="MedlineCitation/Article/Journal/JournalIssue/PubDate/Month"/>
</xsl:template>

</xsl:stylesheet>
