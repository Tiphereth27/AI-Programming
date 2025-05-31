from dataclasses import dataclass
from typing import List
from .time_slot import TimeSlot
from datetime import time

@dataclass
class PreferredTimeRange:
    day: int
    start_time: time
    end_time: time

    def __post_init__(self):
        if not 0 <= self.day <= 4:
            raise ValueError("요일은 0(월)부터 4(금)까지여야 합니다.")
        if self.start_time >= self.end_time:
            raise ValueError("시작 시간은 종료 시간보다 빨라야 합니다.")
        
        if not (time(9, 0) <= self.start_time <= time(18, 0) and 
                time(9, 0) <= self.end_time <= time(18, 0)):
            raise ValueError("수업 시간은 9시부터 18시 사이여야 합니다.")
        
        duration = (self.end_time.hour - self.start_time.hour) + \
                  (self.end_time.minute - self.start_time.minute) / 60
        if not 1 <= duration <= 3:
            raise ValueError("수업 시간은 1시간에서 3시간 사이여야 합니다.")

@dataclass
class UserPreferences:
    min_credits: int
    max_credits: int
    preferred_days: List[int]
    preferred_times: List[PreferredTimeRange]
    excluded_courses: List[str]
    preferred_rating: float = 3.0
    must_take_courses: List[str] = None

    def __post_init__(self):
        # 학점 범위 검사
        if not 1 <= self.min_credits <= self.max_credits <= 21:
            raise ValueError("학점 범위가 올바르지 않습니다.")
        
        # 선호 요일 검사
        if not self.preferred_days:
            raise ValueError("선호 요일을 최소 하나 이상 선택해야 합니다.")
        if not all(0 <= day <= 4 for day in self.preferred_days):
            raise ValueError("선호 요일은 0(월)부터 4(금)까지여야 합니다.")
        if len(set(self.preferred_days)) != len(self.preferred_days):
            raise ValueError("선호 요일에 중복이 있습니다.")
        
        # 선호 시간대 검사
        if not self.preferred_times:
            raise ValueError("선호 시간대를 최소 하나 이상 선택해야 합니다.")
        for time_range in self.preferred_times:
            if not isinstance(time_range, PreferredTimeRange):
                raise ValueError("선호 시간대는 PreferredTimeRange 객체여야 합니다.")
        
        # 제외 과목 검사
        if not isinstance(self.excluded_courses, list):
            raise ValueError("제외 과목은 리스트여야 합니다.")
        
        # 필수 수강 과목 검사
        if self.must_take_courses is None:
            self.must_take_courses = []
        if not isinstance(self.must_take_courses, list):
            raise ValueError("필수 수강 과목은 리스트여야 합니다.")
        
        # 선호 평점 검사
        if not 0 <= self.preferred_rating <= 5:
            raise ValueError("선호 평점은 0부터 5 사이여야 합니다.") 