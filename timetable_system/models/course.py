from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import time
from .time_slot import TimeSlot

class CourseError(Exception):
    """과목 관련 에러"""
    pass

class CourseType:
    """과목 유형 정의"""
    MAJOR_REQUIRED = "전공필수"
    MAJOR_ELECTIVE = "전공선택"

    @classmethod
    def get_valid_types(cls) -> List[str]:
        """유효한 과목 유형 목록 반환"""
        return [
            cls.MAJOR_REQUIRED,
            cls.MAJOR_ELECTIVE,
        ]

@dataclass
class Course:
    code: str
    name: str
    credits: int
    time_slots: List[TimeSlot]
    rating: float = 3.0
    course_type: str = CourseType.MAJOR_REQUIRED
    year: int = 1
    professor: Optional[str] = None
    description: Optional[str] = None

    def __post_init__(self):
        # 기본 필드 검사
        if not isinstance(self.code, str) or not self.code:
            raise CourseError("과목 코드는 비어있지 않은 문자열이어야 합니다.")
        if not isinstance(self.name, str) or not self.name:
            raise CourseError("과목명은 비어있지 않은 문자열이어야 합니다.")
        
        # 학점 검사
        if not isinstance(self.credits, int):
            raise CourseError("학점은 정수여야 합니다.")
        if not 1 <= self.credits <= 3:
            raise CourseError("학점은 1~3 사이여야 합니다.")
        
        # 시간 슬롯 검사
        if not isinstance(self.time_slots, list):
            raise CourseError("시간 슬롯은 리스트여야 합니다.")
        if not self.time_slots:
            raise CourseError("수업 시간이 지정되어야 합니다.")
        if not all(isinstance(slot, TimeSlot) for slot in self.time_slots):
            raise CourseError("시간 슬롯은 TimeSlot 객체여야 합니다.")
        
        # 평점 검사
        if not isinstance(self.rating, (int, float)):
            raise CourseError("평점은 숫자여야 합니다.")
        if not 0 <= self.rating <= 5:
            raise CourseError("평점은 0부터 5 사이여야 합니다.")
        
        # 과목 유형 검사
        if not isinstance(self.course_type, str):
            raise CourseError("과목 유형은 문자열이어야 합니다.")
        if self.course_type not in CourseType.get_valid_types():
            raise CourseError(f"유효하지 않은 과목 유형입니다. 유효한 유형: {CourseType.get_valid_types()}")
        
        # 학년 검사
        if not isinstance(self.year, int):
            raise CourseError("학년은 정수여야 합니다.")
        if not 1 <= self.year <= 4:
            raise CourseError("학년은 1에서 4 사이여야 합니다.")

    def get_total_hours(self) -> float:
        """총 수업 시간(시간 단위)을 반환"""
        return sum(slot.get_duration() for slot in self.time_slots)

    def has_time_conflict(self, other: 'Course') -> bool:
        """다른 과목과 시간이 충돌하는지 확인"""
        if not isinstance(other, Course):
            raise CourseError("비교 대상은 Course 객체여야 합니다.")
        
        for slot1 in self.time_slots:
            for slot2 in other.time_slots:
                if slot1.overlaps_with(slot2):
                    return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Course 객체를 딕셔너리로 변환"""
        return {
            "code": self.code,
            "name": self.name,
            "credits": self.credits,
            "time_slots": [slot.to_dict() for slot in self.time_slots],
            "rating": self.rating,
            "course_type": self.course_type,
            "year": self.year,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Course':
        """딕셔너리에서 Course 객체 생성"""
        try:
            time_slots = []
            for slot in data['time_slots']:
                time_slots.append(TimeSlot.from_dict(slot))
            
            return cls(
                code=data['code'],
                name=data['name'],
                credits=data['credits'],
                time_slots=time_slots,
                rating=data.get('rating', 3.0),
                course_type=data.get('course_type', CourseType.MAJOR_REQUIRED),
                year=data.get('year', 1),
            )
        except KeyError as e:
            raise CourseError(f"필수 필드가 누락되었습니다: {str(e)}")
        except ValueError as e:
            raise CourseError(f"데이터 형식이 올바르지 않습니다: {str(e)}")

@dataclass
class UserPreferences:
    """사용자 선호도 설정"""
    min_credits: int
    max_credits: int
    preferred_days: List[int]
    preferred_times: List['PreferredTimeRange']
    excluded_courses: List[str] = None
    must_take_courses: List[str] = None
    preferred_rating: float = 3.0

    def __post_init__(self):
        if self.excluded_courses is None:
            self.excluded_courses = []
        if self.must_take_courses is None:
            self.must_take_courses = []

@dataclass
class PreferredTimeRange:
    """선호 시간대"""
    day: int
    start_time: time
    end_time: time 