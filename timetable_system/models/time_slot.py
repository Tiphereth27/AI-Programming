from datetime import time
from dataclasses import dataclass
from typing import Optional

class TimeSlotError(Exception):
    """시간 슬롯 관련 에러"""
    pass

@dataclass
class TimeSlot:
    day: int
    start_time: time
    end_time: time

    def __post_init__(self):
        # 요일 검사
        if not 0 <= self.day <= 4:
            raise TimeSlotError("요일은 0(월)부터 4(금)까지여야 합니다.")
        
        # 시간 형식 검사
        if not isinstance(self.start_time, time) or not isinstance(self.end_time, time):
            raise TimeSlotError("시작 시간과 종료 시간은 time 객체여야 합니다.")
        
        # 시작/종료 시간 검사
        if self.start_time >= self.end_time:
            raise TimeSlotError("시작 시간은 종료 시간보다 빨라야 합니다.")
        
        # 수업 시간 범위 검사
        if not (time(9, 0) <= self.start_time <= time(18, 0) and 
                time(9, 0) <= self.end_time <= time(18, 0)):
            raise TimeSlotError("수업 시간은 9시부터 18시 사이여야 합니다.")
        
        # 수업 길이 검사
        duration = self.get_duration()
        if not 1 <= duration <= 3:
            raise TimeSlotError("수업 시간은 1시간에서 3시간 사이여야 합니다.")

    def get_duration(self) -> float:
        """수업 시간(시간 단위)을 반환"""
        return (self.end_time.hour - self.start_time.hour) + \
               (self.end_time.minute - self.start_time.minute) / 60

    def overlaps_with(self, other: 'TimeSlot') -> bool:
        """다른 시간 슬롯과 겹치는지 확인"""
        if not isinstance(other, TimeSlot):
            raise TimeSlotError("비교 대상은 TimeSlot 객체여야 합니다.")
        
        if self.day != other.day:
            return False
            
        return (self.start_time < other.end_time and 
                self.end_time > other.start_time)

    def to_dict(self) -> dict:
        """TimeSlot 객체를 딕셔너리로 변환"""
        return {
            "day": self.day,
            "start_time": self.start_time.strftime('%H:%M'),
            "end_time": self.end_time.strftime('%H:%M')
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TimeSlot':
        """딕셔너리에서 TimeSlot 객체 생성"""
        try:
            start_hour, start_min = map(int, data['start_time'].split(':'))
            end_hour, end_min = map(int, data['end_time'].split(':'))
            
            return cls(
                day=data['day'],
                start_time=time(start_hour, start_min),
                end_time=time(end_hour, end_min)
            )
        except (KeyError, ValueError) as e:
            raise TimeSlotError(f"잘못된 데이터 형식: {str(e)}") 