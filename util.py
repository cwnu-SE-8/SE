from bs4 import BeautifulSoup

def link_page(page_num: int) -> str:
    return f"https://www.aihub.or.kr/aihubdata/data/list.do?pageIndex={page_num}&currMenu=115&topMenu=100&dataSetSn=&srchdataClCode=DATACL001&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20"
    # return f"https://www.aihub.or.kr/aihubdata/data/list.do?pageIndex={page_num}&currMenu=115&topMenu=100&dataSetSn=&srchdataClCode=DATACL001&srchDataRealmCode=REALM015&searchKeyword=&srchDetailCnd=DETAILCND001&srchOrder=ORDER001&srchPagePer=20"

from bs4 import BeautifulSoup


def html_table_to_markdown(html: str) -> str:
    """
    HTML Table구조를 markdown 형식을 표로 바꾸는 함수

    :param html: 변환할 HTML 태그
    :return: markdown 표 문자열
    """
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table')

    if not table:
        return "No table found in HTML"

    # 모든 행과 열 정보를 저장할 2차원 리스트 초기화
    rows = []

    # rowspan 정보를 저장할 딕셔너리
    rowspan_data = {}

    # 테이블의 모든 행을 처리
    for row_idx, tr in enumerate(table.find_all('tr')):
        current_row = []
        col_idx = 0

        # rowspan으로 인해 이전 행에서 확장된 셀 처리
        while col_idx in rowspan_data and row_idx < rowspan_data[col_idx]['end_row']:
            current_row.append(rowspan_data[col_idx]['value'])
            col_idx += 1

        # 현재 행의 셀들을 처리
        for td in tr.find_all(['td', 'th']):
            # colspan 처리
            colspan = int(td.get('colspan', 1))
            # rowspan 처리
            rowspan = int(td.get('rowspan', 1))

            # 셀 내용 가져오기
            cell_content = td.get_text().strip()

            # rowspan이 1보다 큰 경우 정보 저장
            if rowspan > 1:
                rowspan_data[col_idx] = {
                    'value': cell_content,
                    'end_row': row_idx + rowspan
                }

            # colspan만큼 셀 내용 반복
            for _ in range(colspan):
                current_row.append(cell_content)
                col_idx += 1

        rows.append(current_row)

    # 모든 열의 최대 너비 계산
    col_widths = []
    for col_idx in range(max(len(row) for row in rows)):
        width = max(len(str(row[col_idx])) if col_idx < len(row) else 0 for row in rows)
        col_widths.append(max(width, 3))  # 최소 3칸 너비 보장

    # 마크다운 테이블 생성
    markdown_lines = []

    # 헤더 행
    if rows:
        header = "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(rows[0], col_widths)) + " |"
        markdown_lines.append(header)

        # 구분선
        separator = "|" + "|".join("-" * (width + 2) for width in col_widths) + "|"
        markdown_lines.append(separator)

        # 데이터 행
        for row in rows[1:]:
            line = "| " + " | ".join(str(cell).ljust(width) for cell, width in zip(row, col_widths)) + " |"
            markdown_lines.append(line)

    return "\n".join(markdown_lines)
